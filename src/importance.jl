@doc raw"""
    struct ImportanceResults
        log_weights :: Array{Float64, 1}
        return_values :: Array{Any, 1}
        traces :: Array{Trace, 1}
    end

Wrapper for results of importance sampling. Implements the following methods 
from `Base`: `getindex`. The log weights are unnormalized. 
"""
struct ImportanceResults
    log_weights :: Array{Float64, 1}
    return_values :: Array{Any, 1}
    traces :: Array{Trace, 1}
end
Base.getindex(r :: ImportanceResults, k) = [t.trace[k].value for t in r.traces]
Base.getindex(r :: ImportanceResults) = r.return_values

@doc raw"""
    importance_results()
Outer constructor for `ImportanceResults`.
"""
importance_results() = ImportanceResults(Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())

@doc raw"""
    function likelihood_weighting(f :: F, params...; nsamples :: Int = 1) where F <: Function

Given a stochastic function f and arguments to the function `params...`, executes `nsamples`
iterations of importance sampling by using the prior as a proposal distribution. The importance
weights are given by ``\log W_n = \ell(t_n)``. Returns an `ImportanceResults` instance. 
"""
function likelihood_weighting(f :: F, params...; nsamples :: Int = 1) where F <: Function
    results = importance_results()
    t = trace()
    for n in 1:nsamples
        r_n = f(t, params...)
        logprob!(t)
        log_w = loglikelihood(t)
        push!(results.log_weights, log_w)
        push!(results.return_values, r_n)
        push!(results.traces, deepcopy(t))
    end
    results
end

@doc raw"""
    function log_evidence(r :: ImportanceResults)

Computes the log evidence (log partition function), 

``
\log Z \equiv \log p(x) \approx -\log N_{\text{samples}} + \log \sum_{n=1}^{N_{\text{samples}}} W_n.
``
"""
function log_evidence(r :: ImportanceResults)
    # Z \approx 1/L \sum W_l 
    # \log Z \approx -\log L + \logsumexp(W)
    -log(length(r.log_weights)) + logsumexp(r.log_weights)
end

@doc raw"""
    function normalized_weights(r :: ImportanceResults)

Computes the normalized weights ``w_n`` from unnormalized weights ``W_n``:

``
w_n = W_n / p(x) = \exp\{\ell(t_n) - \log Z\}.
``
"""
function normalized_weights(r :: ImportanceResults)
    log_normalizer = log_evidence(r)
    exp.(r.log_weights .- log_normalizer)
end

@doc raw"""
    function sample(r :: ImportanceResults, k, n :: Int)

Draws `n` samples from the empirical marginal posterior at address `k`.
"""
function sample(r :: ImportanceResults, k, n :: Int)
    v = r[k]
    weights = StatsBase.Weights(normalized_weights(r))
    StatsBase.sample(v, weights, n)
end

export likelihood_weighting, log_evidence, sample