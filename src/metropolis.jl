abstract type Proposal end

struct Symmetric <: Proposal end
struct Prior <: Proposal end
struct Asymmetric <: Proposal end

const SYMMETRIC = Symmetric()
const PRIOR = Prior()
const ASYMMETRIC = Asymmetric()

function sample(t :: Trace, a, d, i :: Proposed)
    new_t = deepcopy(t)
    prop = sample(new_t, a, d, NONSTANDARD)
    (prop, new_t)
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

function log_acceptance_ratio(t :: Trace, t_proposed :: Trace, p :: Symmetric)
    log_proposal_joint = logprob(t_proposed)
    log_orig_joint = logprob(t)
    log_proposal_joint - log_orig_joint
end

function log_acceptance_ratio(t :: Trace, t_proposed :: Trace, p :: Prior)
    log_proposal_lik = loglikelihood(t_proposed)
    log_orig_lik = loglikelihood(t)
    log_proposal_lik - log_orig_lik
end

function mh_step(t :: Trace, f :: F, params...) where F <: Function 
    new_t = trace()
    f(new_t, params...)
    log_a = log_acceptance_ratio(t, new_t, PRIOR)
    if log(rand()) < log_a
        return new_t
    else
        return t
    end
end

function mh_step(f :: F1, q :: F2) where {F1, F2 <: Function}

end

gaussian_single_site_proposal(t :: Trace, a; scale :: Float64 = 0.5) = propose(t, a, Normal(t[a].value, scale))
symmetric(f::typeof(gaussian_single_site_proposal)) = true


export propose, is_symmetric, sampled_addresses
export gaussian_single_site_proposal, mh_step