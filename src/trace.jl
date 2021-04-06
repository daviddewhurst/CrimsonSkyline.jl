const Maybe{T} = Union{T, Nothing}

abstract type Interpretation end
struct Nonstandard <: Interpretation end
struct Standard <: Interpretation end
struct Replayed <: Interpretation end
struct Conditioned <: Interpretation end 
struct Deterministic <: Interpretation end

mutable struct Node{A, D, T, P}
    address :: A
    dist :: D
    value :: Maybe{T}
    logprob :: P
    logprob_sum :: Float64
    observed :: Bool
    pa :: Array{Node, 1}
    ch :: Array{Node, 1}
    interpretation :: Interpretation
end

function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    Node{A, D, T, Float64}(address, dist, nothing, 0.0, 0.0, is_obs, Array{Node, 1}(), Array{Node, 1}(), i)
end

function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    T = typeof(value)
    lp = logpdf.(dist, value)
    P = typeof(lp)
    Node{A, D, T, P}(address, dist, value, lp, sum(lp), is_obs, Array{Node, 1}(), Array{Node, 1}(), i)
end

mutable struct Trace
    trace :: OrderedDict{Any, Node}
    logprob_sum :: Float64
end

trace() = Trace(OrderedDict{Any, Node}(), 0.0)
Base.setindex!(t :: Trace, k, v) = setindex!(t.trace, k, v)
Base.getindex(t :: Trace, k) = t.trace[k]
Base.values(t :: Trace) = values(t.trace)
Base.keys(t :: Trace) = keys(t.trace)
Base.length(t :: Trace) = length(t.trace)

function logprob(t :: Trace)
    l = 0.0
    for v in values(t)
        l += v.logprob_sum
    end
    l
end

function logprob!(t :: Trace)
    l = logprob(t)
    t.logprob_sum = l
end

function loglikelihood(t :: Trace)
    l = 0.0
    for v in values(t)
        if v.observed
            l += v.logprob_sum
        end
    end
    l
end

function pa_from_trace(t :: Trace, pa)
    this_pa = Array{Node, 1}()
    for that_address in pa
        append!(this_pa, (t[that_address],))
    end
    this_pa
end

function connect_pa_ch!(t :: Trace, pa, a)
    append!(t[a].pa, pa_from_trace(t, pa))
    for that_n in t[a].pa
        append!(that_n.ch, (t[a],))
    end
end

function sample(t :: Trace, a, d, i :: Nonstandard; pa = ())
    s = rand(d)
    n = node(s, a, d, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    s
end

function sample(t :: Trace, a, d, i :: Replayed; pa = ())
    n = node(t[a].value, a, d, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    t[a].value
end

function sample(t :: Trace, a, d, params, i :: Replayed; pa = ())
    sample(t, a, d, i; pa = pa)
end

function sample(t :: Trace, a, d, params, i :: Nonstandard; pa = ())
    s = rand(d, params...)
    n = node(s, a, d, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    s
end

function sample(t :: Trace, a, d, s, i :: Standard; pa = ())
    n = node(s, a, d, true, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    s
end

function sample(t :: Trace, a, f, v, i :: Deterministic; pa = ())
    T = typeof(v)
    r = f(v...)
    n = node(T, a, f, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    r
end

function transform(t :: Trace, a, f :: F, v; pa = ()) where F <: Function
    sample(t, a, f, v, Deterministic(); pa = pa)
end

function sample(t :: Trace, a, d; pa = ())
    if a in keys(t)
        sample(t, a, d, t[a].interpretation; pa = pa)
    else
        sample(t, a, d, Nonstandard(); pa = pa)
    end
end

function sample(t :: Trace, a, d, params; pa = ())
    if a in keys(t)
        sample(t, a, d, params, t[a].interpretation; pa = pa)
    else
        sample(t, a, d, params, Nonstandard(); pa = pa)
    end
end

function observe(t :: Trace, a, d, s; pa = ())
    sample(t, a, d, s, Standard(); pa = pa)
end

function prior(f :: F, addresses :: Union{AbstractArray, Tuple}, params...; nsamples :: Int = 1) where F <: Function
    d = Dict(a => [] for a in addresses)
    t = trace()
    for n in 1:nsamples
        f(t, params...)
        kk = keys(t)
        for a in addresses
            if a in kk
                push!(d[a], t[a].value)
            else
                push!(d[a], nothing)
            end
        end
    end
    d
end


export Node, node, Trace, trace, logprob, logprob!, sample, observe, loglikelihood, prior
export node_info, graph, transform
export Interpretation, Nonstandard, Standard, Replayed, Deterministic