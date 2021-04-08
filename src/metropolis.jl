abstract type Proposal end

struct Symmetric <: Proposal end
struct Prior <: Proposal end
struct Asymmetric <: Proposal end

const SYMMETRIC = Symmetric()
const PRIOR = Prior()
const ASYMMETRIC = Asymmetric()

function sample(t :: Trace, a, d, i :: Proposed)
    new_t = deepcopy(t)
    sample(new_t, a, d, NONSTANDARD)
    new_t
end
propose(t :: Trace, a, d) = sample(t, a, d, PROPOSED)
symmetric(f) = false

function loglatent(t :: Trace)
    l = 0.0
    for v in values(t)
        if !v.observed
            l += v.logprob_sum
        end
    end
    l
end

function log_acceptance_ratio(t :: Trace, t_proposed :: Trace, p :: Prior)
    log_proposal_lik = loglikelihood(t_proposed)
    log_orig_lik = loglikelihood(t)
    log_proposal_lik - log_orig_lik
end

function accept(t :: Trace, new_t :: Trace, log_a :: Float64)
    if log(rand()) < log_a
        new_t
    else
        t
    end
end

function mh_step(t :: Trace, f :: F; params = ()) where F <: Function 
    new_t = trace()
    f(new_t, params...)
    log_a = log_acceptance_ratio(t, new_t, PRIOR)
    accept(t, new_t, log_a)
end



export propose, is_symmetric, sampled_addresses
export mh_step