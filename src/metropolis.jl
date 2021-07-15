abstract type Proposal end

struct Symmetric <: Proposal end
struct Prior <: Proposal end
struct Asymmetric <: Proposal end

const SYMMETRIC = Symmetric()
const PRIOR = Prior()
const ASYMMETRIC = Asymmetric()

struct Metropolis <: InferenceType end
const METROPOLIS = Metropolis()

bare_metropolis_results() = BareResults{Metropolis}(METROPOLIS, DefaultDict{Any, Vector{Any}}([]))
metropolis_results() = NonparametricSamplingResults{Metropolis}(METROPOLIS, Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())
metropolis_results(A::DataType, D::DataType) = NonparametricSamplingResults{Metropolis}(METROPOLIS, Array{Float64, 1}(), Array{Any, 1}(), Array{TypedTrace{A, D}, 1}())
symmetric(f :: F) where F <: Function = false

### independent prior metropolis stuff ###

@doc raw"""
    function loglatent(t :: Trace)

Computes the joint log probability of all latent variables in a trace, ``\log p(t) - \ell(t)``.
"""
function loglatent(t :: Trace)
    l = 0.0
    for v in values(t)
        if !v.observed
            l += v.logprob_sum
        end
    end
    l
end

@doc raw"""
    function log_acceptance_ratio(t :: Trace, t_proposed :: Trace, p :: Prior)

Computes the log acceptance ratio of a Metropolis step when using the independent prior proposal 
algorithm:

``
\log \alpha = \ell(t_{\text{proposed}}) - \ell(t_{\text{original}})
``
"""
function log_acceptance_ratio(t :: Trace, t_proposed :: Trace, p :: Prior)
    log_proposal_lik = loglikelihood(t_proposed)
    log_orig_lik = loglikelihood(t)
    log_proposal_lik - log_orig_lik
end

@doc raw"""
    function accept(t :: Trace, new_t :: Trace, log_a :: Float64)

Stochastic function that either returns `new_t` if accepted or returns `t` if not accepted.
"""
accept(t :: Trace, new_t :: Trace, log_a :: Float64) = log(rand()) < log_a ? new_t : t

@doc raw"""
    function mh_step(t :: Trace, f; params = (), return_val :: Bool = false)

An independent prior sample Metropolis step.

Given a trace `t` and stochatic function `f` depending on `params...`, generates proposals 
from prior draws and accepts based on the likelihood ratio.
"""
function mh_step(t :: Trace, f; params = (), return_val :: Bool = false)
    new_t = trace()
    r = f(new_t, params...)
    log_a = log_acceptance_ratio(t, new_t, PRIOR)
    a = accept(t, new_t, log_a)
    if !return_val
        a
    else
        (r, a)
    end
end

@doc raw"""
    function mh_step(t :: Trace, f, types::Tuple{DataType, DataType}; params = (), return_val :: Bool = false)

An independent prior sample Metropolis step.

Given a trace `t` and stochastic function `f` depending on `params...`, generates proposals 
from prior draws and accepts based on the likelihood ratio.
"""
function mh_step(t :: Trace, f, types::Tuple{DataType, DataType}; params = (), return_val :: Bool = false)
    new_t = trace(types[1], types[2])
    r = f(new_t, params...)
    log_a = log_acceptance_ratio(t, new_t, PRIOR)
    a = accept(t, new_t, log_a)
    if !return_val
        a
    else
        (r, a)
    end
end

@doc raw"""
    function mh(f; params = (), burn = 1000, thin = 50, num_iterations = 10000)

Generic Metropolis algorithm using draws from the prior.

Args:

+ `f`: stochastic function. Must have call signature `f(t :: Trace, params...)`
+ `params`: addditional arguments to pass to `f` and each of the proposal kernels.
+ `burn`: number of samples to discard at beginning of markov chain
+ `thin`: keep only every `thin`-th draw. E.g., if `thin = 100`, only every 100-th trace will be kept.
+ `num_iterations`: total number of steps to take in the markov chain
"""
function mh(f; params = (), burn = 1000, thin = 50, num_iterations = 10000)
    results = metropolis_results()
    t = trace()
    r = f(t, params...)
    for n in 1:num_iterations
        r, t = mh_step(t, f; params = params, return_val = true)
        if (n > burn) && (n % thin == 0)
            push!(results.return_values, r)
            push!(results.traces, t)
            push!(results.log_weights, logprob(t))
        end
    end
    results
end

@doc raw"""
    function mh(f, types::Tuple{DataType, DataType}; params = (), burn = 1000, thin = 50, num_iterations = 10000)

Generic Metropolis algorithm using draws from the prior.

Args:

+ `f`: stochastic function. Must have call signature `f(t :: Trace, params...)`
+ `params`: addditional arguments to pass to `f` and each of the proposal kernels.
+ `burn`: number of samples to discard at beginning of markov chain
+ `thin`: keep only every `thin`-th draw. E.g., if `thin = 100`, only every 100-th trace will be kept.
+ `num_iterations`: total number of steps to take in the markov chain
"""
function mh(f, types::Tuple{DataType, DataType}; params = (), burn = 1000, thin = 50, num_iterations = 10000)
    results = metropolis_results(types...)
    t = trace(types...)
    r = f(t, params...)
    for n in 1:num_iterations
        r, t = mh_step(t, f, types; params = params, return_val = true)
        if (n > burn) && (n % thin == 0)
            push!(results.return_values, r)
            push!(results.traces, t)
            push!(results.log_weights, logprob(t))
        end
    end
    results
end

function sample(r :: NonparametricSamplingResults{Metropolis}, k, n :: Int)
    v = r[k]
    if n == 1
        StatsBase.sample(v)
    else
        StatsBase.sample(v, n)
    end
end

function sample(r :: ParametricSamplingResults{Metropolis}, k, n :: Int)
    v = getsampled(r, k)
    if n == 1
        StatsBase.sample(v)
    else
        StatsBase.sample(v, n)
    end
end

### general metropolis stuff ###

@doc raw"""
    function logprob(t0 :: Trace, t1 :: Trace)  

Computes the proposal log probability ``q(t_1 | t_0)``.

This expression has two parts: log probability that is generated at the proposed site(s), and 
log probability that is generated at the sites that are present in `t1` but not in `t0`. 
"""
function logprob(t0 :: Trace, t1 :: Trace)
    log_q = 0.0
    kt0 = keys(t0)
    kt1 = keys(t1)
    # get lp of new addresses in domain
    new_addresses = setdiff(kt1, kt0)  # 0 for static structure
    for a in new_addresses
        log_q += t1[a].logprob_sum
    end
    # get lp of addresses selected by proposal
    common_addresses = intersect(kt0, kt1)
    for a in common_addresses
        if t0[a].value != t1[a].value
            log_q += t1[a].logprob_sum
        end
    end
    log_q
end

@doc raw"""
    function copy_common!(old_t :: Trace, new_t :: Trace)

Copies nodes from `old_t` into `new_t` for all addresses in the intersection
of their address sets. 
"""
function copy_common!(old_t :: Trace, new_t :: Trace)
    common_addresses = intersect(keys(old_t), keys(new_t))
    for a in common_addresses
        new_t[a] = old_t[a]
    end
    new_t
end

# proposal signature
# function q(old_trace :: Trace, new_trace :: Trace, params...)

@doc raw"""
    function mh_step(t :: Trace, f, q; params = (), return_val :: Bool = false)

A generic Metropolis step using an arbitrary proposal kernel. 

Given a trace `t`, a stochastic function `f` with signature `f(t :: Trace, params...)` a stochastic function 
`q` with signature `q(old_trace :: Trace, new_trace :: Trace, params...)`, generates a proposal from `q` and 
accepts based on the log acceptance probability:

``
\log \alpha = \log p(t_{\text{new}}) - \log q(t_{\text{new}}|t_{\text{old}}) - [\log p(t_{\text{old}}) - \log q(t_{\text{old}} | t_{\text{new}})].
``
"""
function mh_step(t :: Trace, f, q; params = (), return_val :: Bool = false)
    new_t = trace()
    f(new_t, params...)
    copy_common!(t, new_t)
    logprob!(t)
    q(t, new_t, params...) # propose into the trace
    new_t, g = replay(f, new_t)
    r = g(params...)  # new joint
    logprob!(new_t)
    log_q_new_given_old = logprob(t, new_t)
    log_q_old_given_new = logprob(new_t, t)
    log_a = new_t.logprob_sum + log_q_old_given_new - t.logprob_sum - log_q_new_given_old
    a = accept(t, new_t, log_a)
    if !return_val
        a
    else
        (r, a)
    end
end

@doc raw"""
    function mh_step(t :: Trace, f, q, types::Tuple{DataType,DataType}; params = (), return_val :: Bool = false)

A generic Metropolis step using an arbitrary proposal kernel. 

Given a trace `t`, a stochastic function `f` with signature `f(t :: Trace, params...)` a stochastic function 
`q` with signature `q(old_trace :: Trace, new_trace :: Trace, params...)`, generates a proposal from `q` and 
accepts based on the log acceptance probability:

``
\log \alpha = \log p(t_{\text{new}}) - \log q(t_{\text{new}}|t_{\text{old}}) - [\log p(t_{\text{old}}) - \log q(t_{\text{old}} | t_{\text{new}})].
``
"""
function mh_step(t :: Trace, f, q, types::Tuple{DataType,DataType}; params = (), return_val :: Bool = false)
    new_t = trace(types[1], types[2])
    f(new_t, params...)
    copy_common!(t, new_t)
    logprob!(t)
    q(t, new_t, params...) # propose into the trace
    new_t, g = replay(f, new_t)
    r = g(params...)  # new joint
    logprob!(new_t)
    log_q_new_given_old = logprob(t, new_t)
    log_q_old_given_new = logprob(new_t, t)
    log_a = new_t.logprob_sum + log_q_old_given_new - t.logprob_sum - log_q_new_given_old
    a = accept(t, new_t, log_a)
    if !return_val
        a
    else
        (r, a)
    end
end

@doc raw"""
    function mh(f, qs :: A; params = (), burn = 100, thin = 10, num_iterations = 10000, inverse_verbosity = 100) where A <: AbstractArray

Generic Metropolis algorithm using user-defined proposal kernels.

Args:

+ `f`: stochastic function. Must have call signature `f(t :: Trace, params...)`
+ `qs`: array-like of proposal kernels. Proposal kernels are applied sequentially in the order that they appear in this array.
    Proposal kernels must have the signature `q(old_t :: Trace, new_t :: Trace, params...)` where it must take in at least the same number of arguments
    in `params` as `f`.
+ `params`: addditional arguments to pass to `f` and each of the proposal kernels.
+ `burn`: number of samples to discard at beginning of markov chain
+ `thin`: keep only every `thin`-th draw. E.g., if `thin = 100`, only every 100-th trace will be kept.
+ `num_iterations`: total number of steps to take in the markov chain
+ `inverse_verbosity`: every `inverse_verbosity` iterations, a stattus report will be logged.
"""
function mh(f, qs :: A; params = (), burn = 100, thin = 10, num_iterations = 10000, inverse_verbosity = 100) where A <: AbstractArray
    results = metropolis_results()
    t = trace()
    f(t, params...)
    local r
    for n in 1:num_iterations
        for q in qs
            (r, t) = mh_step(t, f, q; params = params, return_val = true)
        end
        if (n > burn) && (n % thin == 0)
            push!(results.return_values, r)
            push!(results.traces, t)
            push!(results.log_weights, logprob(t))
        end
        if n % inverse_verbosity == 0
            @info "On iteration $n of $num_iterations"
        end
    end
    results
end

@doc raw"""
    function mh(f, qs :: A, addresses; params = (), burn = 100, thin = 10, num_iterations = 10000, inverse_verbosity = 100) where A <: AbstractArray

Generic Metropolis algorithm using user-defined proposal kernels, returning only a requested subset of 
addresses. 

Args:

+ `f`: stochastic function. Must have call signature `f(t :: Trace, params...)`
+ `qs`: array-like of proposal kernels. Proposal kernels are applied sequentially in the order that they appear in this array.
    Proposal kernels must have the signature `q(old_t :: Trace, new_t :: Trace, params...)` where it must take in at least the same number of arguments
    in `params` as `f`.
+ `addresses`: *only* values sampled at these addresses will be saved in the `values` field of the
    `BareResults` struct returned.
+ `params`: addditional arguments to pass to `f` and each of the proposal kernels.
+ `burn`: number of samples to discard at beginning of markov chain
+ `thin`: keep only every `thin`-th draw. E.g., if `thin = 100`, only every 100-th trace will be kept.
+ `num_iterations`: total number of steps to take in the markov chain
+ `inverse_verbosity`: every `inverse_verbosity` iterations, a stattus report will be logged.
"""
function mh(f, qs :: A, addresses; params = (), burn = 100, thin = 10, num_iterations = 10000, inverse_verbosity = 100) where A <: AbstractArray
    results = bare_metropolis_results()
    t = trace()
    f(t, params...)
    local r
    for n in 1:num_iterations
        for q in qs
            (r, t) = mh_step(t, f, q; params = params, return_val = true)
        end
        if (n > burn) && (n % thin == 0)
            for a in addresses
                v = a in keys(t) ? t[a].value : nothing
                push!(results.values[a], v)
            end
        end
        if n % inverse_verbosity == 0
            @info "On iteration $n of $num_iterations"
        end
    end
    results
end

@doc raw"""
    function mh(f, qs :: A, types::Tuple{DataType,DataType}; params = (), burn = 100, thin = 10, num_iterations = 10000, inverse_verbosity = 100) where A <: AbstractArray

Generic Metropolis algorithm using user-defined proposal kernels.

Args:

+ `f`: stochastic function. Must have call signature `f(t :: Trace, params...)`
+ `qs`: array-like of proposal kernels. Proposal kernels are applied sequentially in the order that they appear in this array.
    Proposal kernels must have the signature `q(old_t :: Trace, new_t :: Trace, params...)` where it must take in at least the same number of arguments
    in `params` as `f`.
+ `params`: addditional arguments to pass to `f` and each of the proposal kernels.
+ `burn`: number of samples to discard at beginning of markov chain
+ `thin`: keep only every `thin`-th draw. E.g., if `thin = 100`, only every 100-th trace will be kept.
+ `num_iterations`: total number of steps to take in the markov chain
+ `inverse_verbosity`: every `inverse_verbosity` iterations, a stattus report will be logged.
"""
function mh(f, qs :: A, types::Tuple{DataType,DataType}; params = (), burn = 100, thin = 10, num_iterations = 10000, inverse_verbosity = 100) where A <: AbstractArray
    results = metropolis_results(types...)
    t = trace(types...)
    f(t, params...)
    local r
    for n in 1:num_iterations
        for q in qs
            (r, t) = mh_step(t, f, q, types; params = params, return_val = true)
        end
        if (n > burn) && (n % thin == 0)
            push!(results.return_values, r)
            push!(results.traces, t)
            push!(results.log_weights, logprob(t))
        end
        if n % inverse_verbosity == 0
            @info "On iteration $n of $num_iterations"
        end
    end
    results
end

export Metropolis
export propose, is_symmetric
export mh_step, mh, metropolis_results