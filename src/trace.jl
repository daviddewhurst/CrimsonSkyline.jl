const Maybe{T} = Union{T, Nothing}

abstract type Interpretation end
struct Nonstandard <: Interpretation end
struct Standard <: Interpretation end
struct Replayed <: Interpretation end
struct Conditioned <: Interpretation end 
struct Blocked <: Interpretation end
struct Deterministic <: Interpretation end
struct Proposed <: Interpretation end

const NONSTANDARD = Nonstandard()
const STANDARD = Standard()
const REPLAYED = Replayed()
const BLOCKED = Blocked()
const CONDITIONED = Conditioned()
const DETERMINISTIC = Deterministic()
const PROPOSED = Proposed()

@doc raw"""
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
        last_interpretation :: Interpretation
    end
"""
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
    last_interpretation :: Interpretation
end

@doc raw"""
    function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}

Outer constructor for `Node` where no data is passed during construction. 
"""
function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    Node{A, D, T, Float64}(address, dist, nothing, 0.0, 0.0, is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

@doc raw"""
    function node(value :: I, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D, I <: AbstractArray}

Outer constructor for `Node` where data is passed during construction. Data type is inferred from the passed data.
"""
function node(value :: I, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D, I <: AbstractArray}
    T = typeof(value)
    lp = logpdf.(dist, value)
    P = typeof(lp)
    Node{A, D, T, P}(address, dist, value, lp, sum(lp), is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

@doc raw"""
    function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}

Outer constructor for `Node` where data is passed during construction. Data type is inferred from the passed data.
"""
function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    T = typeof(value)
    lp = logpdf(dist, value)
    P = typeof(lp)
    Node{A, D, T, P}(address, dist, value, lp, sum(lp), is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

@doc raw"""
    mutable struct Trace
        trace :: OrderedDict{Any, Node}
        logprob_sum :: Float64
    end

`Trace`s support the following `Base` methods: `setindex!`, `getindex`,
`keys`, `values`, and `length`.
"""
mutable struct Trace
    trace :: OrderedDict{Any, Node}
    logprob_sum :: Float64
end

@doc raw"""
    trace()

This is the recommended way to construct a new trace.
"""
trace() = Trace(OrderedDict{Any, Node}(), 0.0)
Base.setindex!(t :: Trace, k, v) = setindex!(t.trace, k, v)
Base.getindex(t :: Trace, k) = t.trace[k]
Base.values(t :: Trace) = values(t.trace)
Base.keys(t :: Trace) = keys(t.trace)
Base.length(t :: Trace) = length(t.trace)

@doc raw"""
    function interpret_latent!(t :: Trace, i :: Interpretation)

Changes the interpretation of all nodes in `t` to have `interpretation == i`
"""
function interpret_latent!(t :: Trace, i :: Interpretation)
    for a in keys(t)
        if !t[a].observed
            t[a].interpretation = i
        end
    end
end

@doc raw"""
    function logprob(t :: Trace)

Computes and returns the joint log probability of the trace:

``\log p(t) = \sum_{a \in \text{keys}(t)}\log p(t[a])``
"""
function logprob(t :: Trace)
    l = 0.0
    for v in values(t)
        l += v.logprob_sum
    end
    l
end

@doc raw"""
    function logprob!(t :: Trace)

Computes the joint log probability to the trace and assigns it to `t.logprob_sum`.
"""
function logprob!(t :: Trace)
    l = logprob(t)
    t.logprob_sum = l
end

@doc raw"""
    function loglikelihood(t :: Trace)

Computes and returns the log likelihood of the observed data under the model:

``
\ell(t) = 
\sum_{a:\ [a \in \text{keys}(t)] \wedge [\text{interpretation}(a) = \text{Standard}]} 
\log p(t[a])
``
"""
function loglikelihood(t :: Trace)
    l = 0.0
    for v in values(t)
        if v.observed
            l += v.logprob_sum
        end
    end
    l
end

@doc raw"""
    function pa_from_trace(t :: Trace, pa)

Collects nodes in trace corresponding to an iterable of parent addresses `pa`.
"""
function pa_from_trace(t :: Trace, pa)
    this_pa = Array{Node, 1}()
    for that_address in pa
        append!(this_pa, (t[that_address],))
    end
    this_pa
end

@doc raw"""
    function connect_pa_ch!(t :: Trace, pa, a)  

Connects parent and child nodes. Adds child nodes to parent's `ch` and 
parent nodes to child's `pa`.
"""
function connect_pa_ch!(t :: Trace, pa, a)
    append!(t[a].pa, pa_from_trace(t, pa))
    for that_n in t[a].pa
        append!(that_n.ch, (t[a],))
    end
end

@doc raw"""
    function sample(t :: Trace, a, d, i :: Nonstandard; pa = ())

Samples from distribution `d` into trace `t` at address `a`.

1. Samples a value from `d`
2. Creates a sample node
3. Adds the sample node to trace `t` at value `a`
4. Optionally adds nodes corresponding to the addresses in `pa` as parent nodes
5. Returns the sampled value
"""
function sample(t :: Trace, a, d, i :: Nonstandard; pa = ())
    s = rand(d)
    n = node(s, a, d, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    s
end

@doc raw"""
    function sample(t :: Trace, a, d, i :: Replayed; pa = ())

Replays the sampled node through the trace. 

1. If `a` is not in `t`'s address set, calls `sample(t, a, d, NONSTANDARD; pa = pa)`. 
2. Creates a sample node that copies the value from the last node stored in the trace at address `a`. 
3. Adds the sample node to trace `t` at value `a`
4. Optionally adds nodes corresponding to the addresses in `pa` as parent nodes
5. Resets the node's interpretation to the original interpretation
"""
function sample(t :: Trace, a, d, i :: Replayed; pa = ())
    if a in keys(t)
        last_i = t[a].last_interpretation
        n = node(t[a].value, a, d, false, i)
        t[a] = n
        connect_pa_ch!(t, pa, a)
        r = t[a].value
        t[a].interpretation = last_i
    else
        r = sample(t, a, d, NONSTANDARD; pa = pa)
    end
    r
end

@doc raw"""
    function sample(t :: Trace, a, d, i :: Blocked; pa = ())

Samples from `d`, deletes the node stored at address `a` from trace `t`, and returns the 
sampled value. 
"""
function sample(t :: Trace, a, d, i :: Blocked; pa = ())
    s = rand(d)
    delete!(t.trace, a)
    s
end

@doc raw"""
    function sample(t :: Trace, a, d, params, i :: Blocked; pa = ())

Samples from `d` passing the optional arguments `params`, deletes the node stored at address `a`
from trace `t`, and returns the sampled value. 
"""
function sample(t :: Trace, a, d, params, i :: Blocked; pa = ())
    s = rand(d, params...)
    delete!(t.trace, a)
    s
end

@doc raw"""
    function sample(t :: Trace, a, d, params, i :: Replayed; pa = ())

Replays the sampled node through the trace. 

1. If `a` is not in `t`'s address set, calls `sample(t, a, d, NONSTANDARD; pa = pa)`. 
2. Creates a sample node that copies the value from the last node stored in the trace at address `a`. 
3. Adds the sample node to trace `t` at value `a`
4. Optionally adds nodes corresponding to the addresses in `pa` as parent nodes
5. Resets the node's interpretation to the original interpretation
"""
function sample(t :: Trace, a, d, params, i :: Replayed; pa = ())
    sample(t, a, d, i; pa = pa)
end

@doc raw"""
    function sample(t :: Trace, a, d, params, i :: Nonstandard; pa = ())

Samples from distribution `d` into trace `t` at address `a`.

1. Samples a value from `d` passing the optional arguments `params`
2. Creates a sample node
3. Adds the sample node to trace `t` at value `a`
4. Optionally adds nodes corresponding to the addresses in `pa` as parent nodes
5. Returns the sampled value
"""
function sample(t :: Trace, a, d, params, i :: Nonstandard; pa = ())
    s = rand(d, params...)
    n = node(s, a, d, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    s
end

@doc raw"""
    function sample(t :: Trace, a, d, s, i :: Standard; pa = ())    

Scores an observed value `s` against the distribution `d`, storing the value in trace `t` at 
address `a` and optionally adds nodes corresponding to the addresses in `pa` as parent nodes.
"""
function sample(t :: Trace, a, d, s, i :: Standard; pa = ())
    n = node(s, a, d, true, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    s
end

@doc raw"""
    function sample(t :: Trace, a, f, v, i :: Deterministic; pa = ())

Creates a deterministic node mapping the tuple of data `v` through function `f`, 
storing the value in trace `t` at address `a`.

1. Infers input type from `v`
2. Maps tuple of data `v` through function `f`, yielding `r = f(v...)`
3. Creates a deterministic node and stores it in `t` at address `a`
4. Optionally adds nodes corresponding to the addresses in `pa` as parent nodes
5. Returns `r`
"""
function sample(t :: Trace, a, f, v, i :: Deterministic; pa = ())
    T = typeof(v)
    r = f(v...)
    n = node(T, a, f, false, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    r
end

@doc raw"""
    function transform(t :: Trace, a, f :: F, v; pa = ()) where F <: Function

Alias for `sample(t, a, f, v, DETERMINISTIC; pa = pa)`.
"""
function transform(t :: Trace, a, f :: F, v; pa = ()) where F <: Function
    sample(t, a, f, v, DETERMINISTIC; pa = pa)
end

@doc raw"""
    function sample(t :: Trace, a, d; pa = ())

If `a` is in the set of trace addresses, calls `sample` using `t[a]`'s interpretation. 
Otherwise, calls `sample` using nonstandard interpretation. 
"""
function sample(t :: Trace, a, d; pa = ())
    if a in keys(t)
        sample(t, a, d, t[a].interpretation; pa = pa)
    else
        sample(t, a, d, NONSTANDARD; pa = pa)
    end
end

@doc raw"""
    function sample(t :: Trace, a, d, params; pa = ())

If `a` is in the set of trace addresses, calls `sample` using `t[a]`'s interpretation. 
Otherwise, calls `sample` using nonstandard interpretation. 
"""
function sample(t :: Trace, a, d, params; pa = ())
    if a in keys(t)
        sample(t, a, d, params, t[a].interpretation; pa = pa)
    else
        sample(t, a, d, params, NONSTANDARD; pa = pa)
    end
end

@doc raw"""
    function observe(t :: Trace, a, d, s; pa = ())

If `d` is not `nothing` an alias for calling `sample` with standard interpretation. 
Otherwise, an alias for calling `sample` with nonstandard interpretation. 
"""
function observe(t :: Trace, a, d, s; pa = ())
    if s === nothing
        sample(t, a, d, NONSTANDARD; pa = pa)
    else
        sample(t, a, d, s, STANDARD; pa = pa)
    end
end

@doc raw"""
    function prior(f :: F, addresses :: Union{AbstractArray, Tuple}, params...; nsamples :: Int = 1) where F <: Function

Given a generative function `f`, an array-like of addresses, and a collection of parameters to
pass to `f`, runs `nsamples` evaluations of the `f`, collecting the values from the `addresses`
and returning a `Dict` mapping addresses to values. 
"""
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
export Nonstandard, Standard, Replayed, Conditioned, Deterministic
export NONSTANDARD, STANDARD, REPLAYED, CONDITIONED, DETERMINISTIC