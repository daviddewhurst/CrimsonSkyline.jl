const Maybe{T} = Union{T, Nothing}

abstract type Interpretation end
struct Nonstandard <: Interpretation end
struct Standard <: Interpretation end
struct Replayed <: Interpretation end
struct Conditioned <: Interpretation end 

mutable struct Node{A, D, T, P}
    address :: A
    dist :: D
    value :: Maybe{T}
    logprob :: P
    logprob_sum :: Float64
    observed :: Bool
    interpretation :: Interpretation
end

function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    Node{A, D, T, Float64}(address, dist, nothing, 0.0, 0.0, is_obs, i)
end

function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    T = typeof(value)
    lp = logpdf.(dist, value)
    P = typeof(lp)
    Node{A, D, T, P}(address, dist, value, lp, sum(lp), is_obs, i)
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

function sample(t :: Trace, a, d, i :: Nonstandard)
    s = rand(d)
    n = node(s, a, d, false, i)
    t[a] = n
    s
end

function sample(t :: Trace, a, d, i :: Replayed)
    n = node(t[a].value, a, d, false, i)
    t[a] = n
    t[a].value
end

function sample(t :: Trace, a, d, params, i :: Replayed)
    sample(t, a, d, i)
end

function sample(t :: Trace, a, d, params, i :: Nonstandard)
    s = rand(d, params...)
    n = node(s, a, d, false, i)
    t[a] = n
    s
end

function sample(t :: Trace, a, d, s, i :: Standard)
    n = node(s, a, d, true, i)
    t[a] = n
    s
end

function sample(t :: Trace, a, d)
    if a in keys(t)
        sample(t, a, d, t[a].interpretation)
    else
        sample(t, a, d, Nonstandard())
    end
end

function sample(t :: Trace, a, d, params)
    if a in keys(t)
        sample(t, a, d, params, t[a].interpretation)
    else
        sample(t, a, d, params, Nonstandard())
    end
end

function observe(t :: Trace, a, d, s)
    sample(t, a, d, s, Standard())
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
export Interpretation, Nonstandard, Standard, Replayed