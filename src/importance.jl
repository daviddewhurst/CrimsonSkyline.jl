struct ImportanceResults
    log_weights :: Array{Float64, 1}
    return_values :: Array{Any, 1}
    traces :: Array{Trace, 1}
end
Base.getindex(r :: ImportanceResults, k) = [t.trace[k].value for t in r.traces]
Base.getindex(r :: ImportanceResults) = r.return_values

function importance_results()
    ImportanceResults(Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())
end

function likelihood_log_weight(t :: Trace)
    log_w = 0.0
    for v in values(t)
        if v.observed
            log_w += v.logprob_sum
        end
    end
    log_w
end

function likelihood_weighting(f :: F, params...; nsamples :: Int = 1) where F <: Function
    results = importance_results()
    for n in 1:nsamples
        t = trace()
        r_n = f(t, params...)
        logprob!(t)
        log_w = likelihood_log_weight(t)
        push!(results.log_weights, log_w)
        push!(results.return_values, r_n)
        push!(results.traces, t)
    end
    results
end

function log_evidence(r :: ImportanceResults)
    # Z \approx 1/L \sum W_l 
    # \log Z \approx -\log L + \logsumexp(W)
    -log(length(r.log_weights)) + logsumexp(r.log_weights)
end

function normalized_weights(r :: ImportanceResults)
    log_normalizer = log_evidence(r)
    exp.(r.log_weights .- log_normalizer)
end

function sample(r :: ImportanceResults, k, n :: Int)
    v = r[k]
    weights = StatsBase.Weights(normalized_weights(r))
    StatsBase.sample(v, weights, n)
end


export likelihood_weighting, log_evidence, sample