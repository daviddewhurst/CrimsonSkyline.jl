const Maybe{T} = Union{T, Nothing}

abstract type Interpretation end
struct Nonstandard <: Interpretation end
struct Standard <: Interpretation end
struct Replayed <: Interpretation end
struct Conditioned <: Interpretation end 
struct Blocked <: Interpretation end
struct Deterministic <: Interpretation end
struct Proposed <: Interpretation end
struct Empirical <: Interpretation end
struct Input <: Interpretation end

const NONSTANDARD = Nonstandard()
const STANDARD = Standard()
const REPLAYED = Replayed()
const BLOCKED = Blocked()
const CONDITIONED = Conditioned()
const DETERMINISTIC = Deterministic()
const PROPOSED = Proposed()
const EMPIRICAL = Empirical()
const INPUT = Input()

abstract type Node end

@doc raw"""
    mutable struct ParametricNode{A, D, T} <: Node
        address :: A
        dist :: D
        value :: Maybe{T}
        logprob :: Float64
        logprob_sum :: Float64
        observed :: Bool
        pa :: Array{Node, 1}
        ch :: Array{Node, 1}
        interpretation :: Union{Interpretation, Vector{Interpretation}}
        last_interpretation :: Union{Interpretation, Vector{Interpretation}}
    end

A `Node` that can be used with arbitrary code for which `rand` and `Distributionss.logpdf` are defined.
"""
mutable struct ParametricNode{A, D, T} <: Node
    address :: A
    dist :: D
    value :: Maybe{T}
    logprob :: Float64
    logprob_sum :: Float64
    observed :: Bool
    pa :: Array{Node, 1}
    ch :: Array{Node, 1}
    interpretation :: Union{Interpretation, Vector{Interpretation}}
    last_interpretation :: Union{Interpretation, Vector{Interpretation}}
end

@doc raw"""
    mutable struct SampleableNode{A, T} <: Node
        address :: A
        dist :: Sampleable
        value :: Maybe{T}
        logprob :: Float64
        logprob_sum :: Float64
        observed :: Bool
        pa :: Array{Node, 1}
        ch :: Array{Node, 1}
        interpretation :: Union{Interpretation, Vector{Interpretation}}
        last_interpretation :: Union{Interpretation, Vector{Interpretation}}
    end

A `Node` that is restricted to be used with any `Sampleable` from `Distributions.jl`.
"""
mutable struct SampleableNode{A, T} <: Node
    address :: A
    dist :: Sampleable
    value :: Maybe{T}
    logprob :: Float64
    logprob_sum :: Float64
    observed :: Bool
    pa :: Array{Node, 1}
    ch :: Array{Node, 1}
    interpretation :: Union{Interpretation, Vector{Interpretation}}
    last_interpretation :: Union{Interpretation, Vector{Interpretation}}
end

@doc raw"""
    function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}

Outer constructor for `Node` where no data is passed during construction. 
"""
function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    ParametricNode{A, D, T}(address, dist, nothing, 0.0, 0.0, is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

@doc raw"""
    function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}

Outer constructor for `Node` where no data is passed during construction. 
"""
function node(T :: DataType, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D <: Sampleable}
    SampleableNode{A, T}(address, dist, nothing, 0.0, 0.0, is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

function setlp!(n::Node, lp) 
    n.logprob = lp
    n.logprob_sum = lp
end

@doc raw"""
    function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}

Outer constructor for `Node` where data is passed during construction. Data type is inferred from the passed data.
"""
function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}
    T = typeof(value)
    lp = logpdf(dist, value)
    ParametricNode{A, D, T}(address, dist, value, lp, lp, is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

@doc raw"""
    function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D}

Outer constructor for `Node` where data is passed during construction. Data type is inferred from the passed data.
"""
function node(value, address :: A, dist :: D, is_obs :: Bool, i :: Interpretation) where {A, D <: Sampleable}
    T = typeof(value)
    lp = logpdf(dist, value)
    SampleableNode{A, T}(address, dist, value, lp, lp, is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

Distributions.logpdf(::Input, value) = 0.0

@doc raw"""
    abstract type Trace end

Base type for all traces. `Trace`s support the following `Base` methods: `setindex!`, `getindex`,
`keys`, `values`, and `length`.
"""
abstract type Trace end

@doc raw"""
    mutable struct UntypedTrace
        trace :: OrderedDict{Any, Node}
        logprob_sum :: Float64
    end

Trace that can hold nodes with all address and value types. 
"""
mutable struct UntypedTrace <: Trace
    trace :: OrderedDict{Any, Node}
    logprob_sum :: Float64
end

@doc raw"""
    mutable struct TypedTrace{A, T} <: Trace
        trace :: OrderedDict{A, SampleableNode{A, T}}
        logprob_sum :: Float64
    end

Trace that can hold nodes of the specific address (`A`) and value (`T`) types.
"""
mutable struct TypedTrace{A, T} <: Trace
    trace :: OrderedDict{A, SampleableNode{A, T}}
    logprob_sum :: Float64
end

@doc raw"""
    trace()

This is the recommended way to construct a new trace.
"""
trace() = UntypedTrace(OrderedDict{Any, Node}(), 0.0)

@doc raw"""
    trace(A, T)

This is the recommended way to construct a new typed trace.
`A` is the address type, `T` is the value type.
"""
trace(A, T) = TypedTrace(OrderedDict{A, SampleableNode{A, T}}(), 0.0)

Base.setindex!(t :: Trace, k, v) = setindex!(t.trace, k, v)
Base.getindex(t :: Trace, k) = t.trace[k]
Base.values(t :: Trace) = values(t.trace)
Base.keys(t :: Trace) = keys(t.trace)
Base.length(t :: Trace) = length(t.trace)

@doc raw"""
    function interpret_latent!(t :: Trace, i :: Interpretation)

Changes the interpretation of all latent nodes in `t` to have `interpretation == i`
"""
function interpret_latent!(t :: Trace, i :: Interpretation)
    for a in keys(t)
        if !t[a].observed && !(t[a].interpretation == CONDITIONED)
            t[a].interpretation = i
        end
    end
end

@doc raw"""
    function logprob(t :: Trace)

Computes and returns the joint log probability of the trace:

```math
\log p(t) = \sum_{a \in \text{keys}(t)}\log p(t[a])
```
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

```math
\ell(t) = 
\sum_{a:\ [a \in \text{keys}(t)] \wedge [\text{interpretation}(a) = \text{Standard}]} 
\log p(t[a])
```
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
    function sample(t :: Trace, a, d, ii :: Array{Interpretation, 1}; pa = ())

Sequentially apply sample statements with interpretations as given in `ii`. This is 
used to depth-first traverse the interpretation graph.
"""
function sample(t :: Trace, a, d, ii :: Vector{Interpretation}; pa = ())
    local s
    for i in ii
        s = sample(t, a, d, i; pa = pa)
    end
    s
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
    function plate(t::Trace, op::F, a, d, s::Int64, i::Nonstandard; pa = ()) where F<:Function

Plate over latent variables.
"""
function plate(t::Trace, op::F, a, d, s::Int64, i::Nonstandard; pa = ()) where F<:Function
    n = node(Vector{eltype(d)}, a, d, false, i)
    rvals = [rand(d) for _ in 1:s]
    lp = 0.0
    for r in rvals
        lp += logpdf(d, r)
    end
    setlp!(n, lp)
    n.value = rvals
    t[a] = n
    connect_pa_ch!(t, pa, a)
    rvals
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
    function plate(t::Trace, op::F, a, d, s::Int64, i::Replayed; pa = ()) where F<:Function

Plate over replayed variables. Note that this method assumes *and does not check* that the value
to be replayed `v` satisfies `length(v) == s`.
"""
function plate(t::Trace, op::F, a, d, s::Int64, i::Replayed; pa = ()) where F<:Function
    n = node(Vector{eltype(d)}, a, d, false, i)
    last_i = t[a].last_interpretation
    lp = 0.0
    for r in t[a].value
        lp += logpdf(d, r)
    end
    setlp!(n, lp)
    n.value = t[a].value
    t[a] = n
    connect_pa_ch!(t, pa, a)
    t[a].interpretation = last_i
    t[a].value
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
    function plate(t::Trace, op::F, a, d, s::Int64, i::Blocked; pa = ()) where F<:Function

Plate over blocked variables.
"""
function plate(t::Trace, op::F, a, d, s::Int64, i::Blocked; pa = ()) where F<:Function
    n = node(Vector{eltype(d)}, a, d, false, i)
    rvals = [rand(d) for _ in 1:s]
    delete!(t.trace, a)
    rvals
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
    function sample(t :: Trace, a, d, params, ii :: Array{Interpretation, 1}; pa = ())

Sequentially apply sample statements with interpretations as given in `ii`. This is 
used to depth-first traverse the interpretation graph.
"""
function sample(t :: Trace, a, d, params, ii :: Array{Interpretation, 1}; pa = ())
    local s
    for i in ii
        s = sample(t, a, d, params, i; pa = pa)
    end
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
    function plate(t::Trace, op::F, a, d, v::Vector{T}; pa = ()) where {T, F<:Function}

Plate over observed variables, i.e., a plated component of model likelihood. `v` is the 
vector of observations, while `op` is likely `observe`.

Example usage: instead of

```
for (i, d) in enumerate(data)
    observe(t, "data $i", Normal(loc, scale), d)
end
```

we can write 

```
plate(t, observe, "data", Normal(loc, scale), data)
```
"""
function plate(t::Trace, op::F, a, d, v::Vector{T}; pa = ()) where {T, F<:Function}
    n = node(typeof(v), a, d, true, STANDARD)
    lp = 0.0
    for r in v
        lp += logpdf(d, r)
    end
    setlp!(n, lp)
    n.value = v
    t[a] = n
    connect_pa_ch!(t, pa, a)
    v
end

@doc raw"""
    function plate(t::Trace, op::F, a, d, v::Vector{T}, params; pa = ()) where {T, F<:Function}

Plate over observed variables with different values but identical distribution, i.e.,
``p(x|z) = \prod_n p(x_n | z_n)``.
This is as opposed to `plate(t::Trace, op::F, a, d, v::Vector{T}; pa = ())`, which is equivalent to
``p(x|z) = \prod_n p(x_n | z)``.

`params` must have the same length as `v`. Each element of `params` corresponds to a vector of that 
particular component of the `params`, i.e., ``z = (z_1, ..., z_D)`` where each ``z_d`` has length ``N``,
the number of observed datapoints, and ``D`` is the cardinality of the parameterization of the 
distribution.

E.g., replace 
```
locs = sample(t, "locs", MvNormal(D, 1.0))
for (i, (loc, d)) in enumerate(zip(locs, data))
    observe(t, "data $i", Normal(loc, 1.0), d)
end
```
with 
```
locs = sample(t, "locs", MvNormal(D, 1.0))
plate(t, observe, "data", Normal, data, (locs, ones(D)))
```
"""
function plate(t::Trace, op::F, a, d, v::Vector{T}, params; pa = ()) where {T, F<:Function}
    # accumulate logprob from values with same distribution
    lp = 0.0
    for (datapoint, param_group) in zip(v, zip(params...))
        lp += logpdf(d(param_group...), datapoint)
    end
    n = node(typeof(v), a, d, true, STANDARD)
    setlp!(n, lp)
    n.value = v
    t[a] = n
    connect_pa_ch!(t, pa, a)
    v
end

@doc raw"""
    function sample(t :: Trace, a, d, i :: Union{Standard,Conditioned}; pa = ())   

Scores an observed value against the distribution `d`, storing the value in trace `t` at 
address `a` and optionally adds nodes corresponding to the addresses in `pa` as parent nodes.

This method is used by the `condition` effect. It will probably not be used by most 
users.
"""
function sample(t :: Trace, a, d, i :: Union{Standard,Conditioned}; pa = ())
    n = node(t[a].value, a, d, true, i)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    t[a].value
end

@doc raw"""
    function plate(t::Trace, op::F, a, d, s::Int64, i::Conditioned; pa = ()) where F<:Function

Plate over conditioned variables.
"""
function plate(t::Trace, op::F, a, d, s::Int64, i::Conditioned; pa = ()) where F<:Function
    n = node(Vector{eltype(d)}, a, d, false, i)
    lp = 0.0
    for r in t[a].value
        lp += logpdf(d, r)
    end
    setlp!(n, lp)
    n.value = t[a].value
    t[a] = n
    connect_pa_ch!(t, pa, a)
    t[a].value
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

function sample(t :: Trace, a, d, i :: Input)
    n = node(d, a, INPUT, true, i)
    t[a] = n
    connect_pa_ch!(t, (), a)
    d
end

@doc raw"""
    input(t :: Trace, a, d)

Track a model input. Used only in graph intermediate representation and factor 
graph.
"""
input(t :: Trace, a, d) = sample(t, a, d, INPUT)

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

# maybe Proposed will have meaning later, stub out just in case
# e.g., right now new trace is made in mh, but is this always true?

sample(t :: Trace, a, d, i :: Proposed) = sample(t, a, d, NONSTANDARD)

@doc raw"""
    propose(t :: Trace, a, d)

Propose a value for the address `a` in trace `t` from the distribution `d`.
"""
propose(t :: Trace, a, d) = sample(t, a, d, PROPOSED)

@doc raw"""
    function plate(t::Trace, op::F, a, d, s::Int64; pa = ()) where F<:Function

Sample or observe a vector of random variables at a single site instead of multiple.
This can speed up inference since the number of sites in the model will no longer scale
with dataset size (though numerical value computation is still linear in dataset size).

Example usage: instead of

```
vals = [sample(t, "val $i", Geometric()) for i in 1:N]
```

we can write 

```
vals = plate(t, sample, "val", Geometric(), N)
```

Mathematically, this is equivalent to the product ``p(z) = \prod_n p(z_n)``
and treating it as the single object ``p(z)`` instead of the ``N`` objects ``p(z_n)``.
"""
function plate(t::Trace, op::F, a, d, s::Int64; pa = ()) where F<:Function
    if a in keys(t)
        plate(t, op, a, d, s, t[a].interpretation; pa = pa)
    else
        plate(t, op, a, d, s, NONSTANDARD; pa = pa)
    end
end
plate(t::Trace, op::F, a, d, v::Vector{Nothing}; pa = ()) where F <: Function = plate(t, op, a, d, length(v); pa = pa)
export plate

function Base.show(io::IO, t::Trace)
    s = "\nTrace with $(length(t)) nodes:"
    for node in values(t)
        s *= "\n$(node.address) = ($(node.value), $(node.dist), $(node.logprob_sum), $(node.interpretation))"
    end
    print(io, s)
end


export Node, ParametricNode, SampleableNode, node
export Trace, UntypedTrace, TypedTrace, trace, logprob, logprob!, sample, observe, input, loglikelihood
export node_info, graph, transform
export Nonstandard, Standard, Replayed, Conditioned, Deterministic, Input, Empirical
export NONSTANDARD, STANDARD, REPLAYED, CONDITIONED, DETERMINISTIC, INPUT, EMPIRICAL