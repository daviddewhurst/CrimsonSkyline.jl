abstract type Importance <: InferenceType end
struct LikelihoodWeighting <: Importance end
struct ImportanceSampling <: Importance end
const LW = LikelihoodWeighting()
const IS = ImportanceSampling()

@doc raw"""
    likelihood_weighting_results()
Outer constructor for `SamplingResults`.
"""
likelihood_weighting_results() = NonparametricSamplingResults{LikelihoodWeighting}(LW, Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())

@doc raw"""
    function lw_step(f :: F, params...) where F <: Function 

Perform one step of likelihood weighting -- draw a single proposal from the prior and compute 
the log weight as equal to the likelihood. Returns a tuple (log weight, rval, trace).
"""
function lw_step(f :: F, params...) where F <: Function 
    t = trace()
    r_n = f(t, params...)
    logprob!(t)
    log_w = loglikelihood(t)
    (log_w, r_n, t)
end

@doc raw"""
    function likelihood_weighting(f :: F, params...; nsamples :: Int = 1) where F <: Function

Given a stochastic function f and arguments to the function `params...`, executes `nsamples`
iterations of importance sampling by using the prior as a proposal distribution. The importance
weights are given by ``\log W_n = \ell(t_n)``. Returns an `SamplingResults` instance. 
"""
function likelihood_weighting(f :: F, params...; nsamples :: Int = 1) where F <: Function
    results = likelihood_weighting_results()
    for n in 1:nsamples
        log_w, r_n, t = lw_step(f, params...)
        push!(results.log_weights, log_w)
        push!(results.return_values, r_n)
        push!(results.traces, deepcopy(t))
    end
    results
end

importance_sampling_results() = NonparametricSamplingResults{ImportanceSampling}(IS, Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())

@doc raw"""
    function is_step(f :: F1, q :: F2; params = ()) where {F1 <: Function, F2 <: Function}

Perform one step of importance sampling -- draw a single sample from the proposal `q`, replay 
it through `f`, and record the log weight as ``\log W_n = \log p(x, z_n) - \log q(z_n)``. Returns
a tuple (log weight, rval, trace). 
"""
function is_step(f :: F1, q :: F2; params = ()) where {F1 <: Function, F2 <: Function}
    proposed_t = trace()
    q(proposed_t, params...)
    logprob!(proposed_t)
    replayed_t, g = replay(f, proposed_t)
    r_n = g(params...)
    logprob!(replayed_t)
    log_w = replayed_t.logprob_sum - proposed_t.logprob_sum
    (log_w, r_n, deepcopy(replayed_t))
end

@doc raw"""
    function importance_sampling(f :: F1, q :: F2; params = (), nsamples :: Int = 1) where {F1 <: Function, F2 <: Function}

Given a stochastic function `f`, a proposal function `q`, and a tuple of `params` to pass to `f` and `q`,
compute `nsamples` iterations of importance sampling. `q` must have the same input signature 
as `f`. Returns a `SamplingResults` instance.
"""
function importance_sampling(f :: F1, q :: F2; params = (), nsamples :: Int = 1) where {F1 <: Function, F2 <: Function}
    results = importance_sampling_results()
    for n in 1:nsamples
        log_w, r_n, t = is_step(f, q; params = params)
        push!(results.log_weights, log_w)
        push!(results.return_values, r_n)
        push!(results.traces, t)
    end
    results
end


@doc raw"""
    function log_evidence(r :: SamplingResults{LikelihoodWeighting})

Computes the log evidence (log partition function), 

``
\log Z \equiv \log p(x) \approx -\log N_{\text{samples}} + \log \sum_{n=1}^{N_{\text{samples}}} W_n.
``
"""
function log_evidence(r :: SamplingResults{I}) where I <: Importance
    # Z \approx 1/L \sum W_l 
    # \log Z \approx -\log L + \logsumexp(W)
    -log(length(r.log_weights)) + logsumexp(r.log_weights)
end

@doc raw"""
    function normalized_weights(r :: SamplingResults{LikelihoodWeighting})

Computes the normalized weights ``w_n`` from unnormalized weights ``W_n``:

``
w_n = W_n / p(x) = \exp\{\ell(t_n) - \log Z\}.
``
"""
function normalized_weights(r :: SamplingResults{I}) where I <: Importance
    log_normalizer = log_evidence(r)
    exp.(r.log_weights .- log_normalizer)
end

@doc raw"""
    function sample(r :: SamplingResults{LikelihoodWeighting}, k, n :: Int)

Draws `n` samples from the empirical marginal posterior at address `k`.
"""
function sample(r :: SamplingResults{I}, k, n :: Int) where I <: Importance
    v = r[k]
    weights = StatsBase.Weights(normalized_weights(r))
    if n == 1
        StatsBase.sample(v, weights)
    else
        StatsBase.sample(v, weights, n)
    end
end

function sample(r :: ParametricSamplingResults{I}, k, n :: Int) where I <: Importance
    v = getsampled(r, k)
    weights = StatsBase.Weights(normalized_weights(r))
    if n == 1
        StatsBase.sample(v, weights)
    else
        StatsBase.sample(v, weights, n)
    end
end

export LikelihoodWeighting, ImportanceSampling
export likelihood_weighting, importance_sampling, lw_step, is_step
export log_evidence, normalized_weights, sample