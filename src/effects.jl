@doc raw"""
    function replay(f :: F, t :: Trace) where F <: Function

Given a stochastic function `f` and a trace `t`, makes `sample` calls behave as though 
they had sampled the values in `t` at the corresponding addresses. 

Returns a tuple `(t_new, g)`, where `t_new` is a trace and `g` is a function.
The function signature of `g` is the same as that of `f` with the first argument removed; that
is, if `f(t :: Trace, params...)`, then `g(params...)`. Computation is delayed, so each of the
latent nodes in `t_new` has `interpretation = REPLAYED`.
Calling `g(params...)` executes the computation and each latent node in `t_new` reverts to 
its original interpretation. 
"""
function replay(f :: F, t :: Trace) where F <: Function
    t_new = deepcopy(t)
    interpret_latent!(t_new, REPLAYED)
    g(params...) = f(t_new, params...)
    (t_new, g)
end

@doc raw"""
    function replace(f :: F, r :: Dict) where F <: Function

Given a mapping `r` from addresses to distribution-like (currently `Distributions`
or `Array{Any, 1}`s), replaces the current distributions at that set of addresses
with this set of distributions. Returns a function `g` that has return signature 
`(t :: Trace, rval)` where `rval` is a return value of `f`.
"""
function replace(f :: F, r :: Dict) where F <: Function
    function g(t :: Trace, params...)
        rval = f(t, params...)
        t = replace(t, r)
        (t, rval)
    end
    g
end

@doc raw"""
    function replace(t :: Trace, r :: Dict)

Given a mapping `r` from addresses to distribution-like (currently `Distributions`
or `Array{Any, 1}`s), replaces the current distributions at that set of addresses
with this set of distributions. Returns the modified trace.
"""
function replace(t :: Trace, r :: Dict)
    for (a, d) in r
        old_n = deepcopy(t[a])
        t[a] = node(old_n.value, old_n.address, d, old_n.observed, old_n.interpretation)
    end
    t
end

@doc raw"""
    function block(f :: F, t :: Trace, addresses) where F <: Function

Given a stochastic function `f`, a trace `t`, and an iterable of addresses, converts traced
randomness into untraced randomness. 

Returns a tuple `(t_new, g)`, where `t_new` is a trace and `g` is a function.
The function signature of `g` is the same as that of `f` with the first argument removed; that
is, if `f(t :: Trace, params...)`, then `g(params...)`. Computation is delayed, so each of 
the latent nodes in `t_new` has `interpretation = BLOCKED`. Calling `g(params...)` executes the 
computation and each latent node in `t_new` with an address in `addresses` is removed.
"""
function block(f :: F, t :: Trace, addresses) where F <: Function
    t_new = deepcopy(t)
    for a in addresses
        t_new[a].interpretation = BLOCKED
    end
    g(params...) = f(t_new, params...)
    (t_new, g)
end

@doc raw"""
    block(f :: F, t :: Trace) where F <: Function

Converts all traced randomness into untraced randomness.
"""
block(f :: F, t :: Trace) where F <: Function = block(f, t, keys(t))

@doc raw"""
    function update(f :: F, r :: SamplingResults{I}) where {F <: Function, I <: InferenceType}

**EXPERIMENTAL**: Given a stochastic function `f` and a `SamplingResults` `r`, update the prior 
predictive to the posterior predictive by jointly replacing all latent sample sites with the joint empirical 
posterior. Returns a stochastic function `g` with the same call signature as `f`. This function will modify in 
place the trace passed into it as the first argument.
"""
function update(f :: F, r :: SamplingResults{I}) where {F <: Function, I <: InferenceType}
    function g(t :: Trace, params...)
        t_sampled = StatsBase.sample(r.traces)
        kts = keys(t_sampled)
        marginals = Dict()
        for a in kts
            if t_sampled[a].interpretation == NONSTANDARD
                sample(t, a, r)
                marginals[a] = r[a]
            end
        end
        new_t, h = replay(f, t)
        rval = h(params...)
        new_t = replace(new_t, marginals)
        (new_t, rval)
    end
    g
end

export block, replay, update, replace