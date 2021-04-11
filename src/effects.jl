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
    function block(f :: F, t :: Trace, addresses) where F <: Function

Given a stoochastic function `f`, a trace `t`, and an iterable of addresses, converts traced
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

export block, replay