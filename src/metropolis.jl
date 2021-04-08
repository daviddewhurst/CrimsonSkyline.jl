abstract type Proposal end

struct Symmetric <: Proposal end
struct Prior <: Proposal end
struct Asymmetric <: Proposal end

const SYMMETRIC = Symmetric()
const PRIOR = Prior()
const ASYMMETRIC = Asymmetric()

# maybe Proposed will have meaning later, stub out just in case
# e.g., right now new trace is made in mh, but is this always true?

sample(t :: Trace, a, d, i :: Proposed) = sample(t, a, d, NONSTANDARD)
propose(t :: Trace, a, d) = sample(t, a, d, PROPOSED)
symmetric(f :: F) where F <: Function = false

### independent prior metropolis stuff ###

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

### general metropolis stuff ###

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

# proposal signature
# function q(old_trace :: Trace, new_trace :: Trace, params...)

function mh_step(t :: Trace, f :: F1, q :: F2; params = ()) where {F1 <: Function, F2 <: Function}
    new_t = deepcopy(t)
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

export propose, is_symmetric
export mh_step