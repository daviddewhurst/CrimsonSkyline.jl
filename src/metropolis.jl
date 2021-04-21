abstract type Proposal end

struct Symmetric <: Proposal end
struct Prior <: Proposal end
struct Asymmetric <: Proposal end

const SYMMETRIC = Symmetric()
const PRIOR = Prior()
const ASYMMETRIC = Asymmetric()

struct Metropolis <: InferenceType end
const METROPOLIS = Metropolis()

metropolis_results() = SamplingResults{Metropolis}(METROPOLIS, Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())

# maybe Proposed will have meaning later, stub out just in case
# e.g., right now new trace is made in mh, but is this always true?

sample(t :: Trace, a, d, i :: Proposed) = sample(t, a, d, NONSTANDARD)

@doc raw"""
    propose(t :: Trace, a, d)

Propose a value for the address `a` in trace `t` from the distribution `d`.
"""
propose(t :: Trace, a, d) = sample(t, a, d, PROPOSED)
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
function accept(t :: Trace, new_t :: Trace, log_a :: Float64)
    if log(rand()) < log_a
        new_t
    else
        t
    end
end

@doc raw"""
    function mh_step(t :: Trace, f :: F; params = ()) where F <: Function 

An independent prior sample Metropolis step.

Given a trace `t` and stochatic function `f` depending on `params...`, generates proposals 
from prior draws and accepts based on the likelihood ratio.
"""
function mh_step(t :: Trace, f :: F; params = (), return_val :: Bool = false) where F <: Function 
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

function mh(f :: F; params = (), burn = 100, thin = 10, num_iterations = 10000) where F <: Function 
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

function sample(r :: SamplingResults{Metropolis}, k, n :: Int)
    v = r[k]
    StatsBase.sample(v, n)
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
    function mh_step(t :: Trace, f :: F1, q :: F2; params = ()) where {F1 <: Function, F2 <: Function}

A generic Metropolis step using an arbitrary proposal kernel. 

Given a trace `t`, a stochastic function `f` with signature `f(t :: Trace, params...)` a stochastic function 
`q` with signature `q(old_trace :: Trace, new_trace :: Trace, params...)`, generates a proposal from `q` and 
accepts based on the log acceptance probability:

``
\log \alpha = \log p(t_{\text{new}}) - \log q(t_{\text{new}}|t_{\text{old}}) - [\log p(t_{\text{old}}) - \log q(t_{\text{old}} | t_{\text{new}})].
``
"""
function mh_step(t :: Trace, f :: F1, q :: F2; params = ()) where {F1 <: Function, F2 <: Function}
    new_t = trace()
    f(new_t, params...)
    copy_common!(t, new_t)
    logprob!(t)
    q(t, new_t, params...) # propose into the trace
    new_t, g = replay(f, new_t)
    g(params...)  # new joint
    logprob!(new_t)
    log_q_new_given_old = logprob(t, new_t)
    log_q_old_given_new = logprob(new_t, t)
    log_a = new_t.logprob_sum + log_q_old_given_new - t.logprob_sum - log_q_new_given_old
    accept(t, new_t, log_a)
end

function mh(f :: F, qs :: A; params = ()) where {F <: Function, A <: AbstractArray}
    error("Not yet implemented!")
end

export propose, is_symmetric
export mh_step, mh