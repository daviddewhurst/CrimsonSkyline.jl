const Maybe{T} = Union{T, Nothing}

mutable struct Node{A, D, T, P}
    address :: A
    dist :: D
    value :: Maybe{T}
    logprob :: P
    logprob_sum :: Float64
end

function node(T :: DataType, address :: A, dist :: D) where {A, D}
    Node{A, D, T, Float64}(address, dist, nothing, 0.0, 0.0)
end

function node(value, address :: A, dist :: D) where {A, D}
    T = typeof(value)
    lp = logpdf.(dist, value)
    P = typeof(lp)
    Node{A, D, T, P}(address, dist, value, lp, sum(lp))
end

mutable struct Trace
    trace :: OrderedDict{Any, Node}
    logprob_sum :: Float64
end

trace() = Trace(OrderedDict{Any, Node}(), 0.0)
Base.setindex!(t :: Trace, k, v) = setindex!(t.trace, k, v)
Base.getindex(t :: Trace, k) = t.trace[k]
Base.values(t :: Trace) = values(t.trace)

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

function sample(t :: Trace, a, d, params...)
    s = rand(d, params...)
    n = node(s, a, d)
    t[a] = n
    s
end


export Node, node, Trace, trace, logprob, logprob!, sample