struct Nested <: InferenceType end
const NESTED = Nested()

nested_results(traces, log_weights) = NonparametricSamplingResults{Nested}(NESTED, log_weights, Array{Any, 1}(), traces)

function sample(r :: SamplingResults{Nested}, k, n :: Int)
    v = r[k]
    log_Z = logsumexp(r.log_weights)
    weights = StatsBase.Weights(exp.(r.log_weights .- log_Z))
    if n == 1
        StatsBase.sample(v, weights)
    else
        StatsBase.sample(v, weights, n)
    end
end

@doc raw"""
    function nested(f, replace_fn; params = (), num_points :: Int64 = 1)

Generic implementation of nested sampling (Skilling, Nested sampling for general Bayesian computation, Bayesian Analysis, 2006).
The number of sampling iterations is a function of `num_points` aka ``N``, and the empirical entropy of the sampling distribution, given at the
``n``-th iteration by ``H_n \approx \sum_k \hat p_k \log \hat p_k^{-1}``, where ``\hat p_k = L_k w_k / Z_k``, ``L_k`` is the likelihood
value, ``w_k`` is the difference in prior mass, and ``Z_k`` is the current estimate of the partition function. The number of sampling iterations
is equal to ``\min_n \{n > 0: n > NH_n\}``. 

Args:
+ `f`: stochastic function. Must have signature `f(t :: Trace, params...)`
+ `replace_fn`: function that returns a tuple `(new_trace :: Trace, new_log_likelihood :: Float64)`. The input signature of this function
    must be `replace_fn(f :: F, log_likelihood :: Float64, params...) where F <: Function`. It must guarantee that `new_log_likelihood > log_likelihood`.
+ `params`: any parameters to pass to `f`
+ `num_points`: the number of likelihood points to keep track of
"""
function nested(f, replace_fn; params = (), num_points :: Int64 = 1)
    traces = [trace() for _ in 1:num_points]
    for t in traces
        f(t, params...)
    end
    loglikelihoods = [loglikelihood(t) for t in traces]
    weights = Float64[]
    pts = Array{Trace, 1}()
    x_last = 1.0
    Z = 0.0  # partition function estimate
    n = 1  # number of iterations

    while true
        min_l_ix = argmin(loglikelihoods)
        log_l_n = loglikelihoods[min_l_ix]

        x_n = exp(-n / num_points)
        w_n = x_last - x_n
        x_last = x_n

        weight_n = log_l_n + log(w_n)
        Z += exp(weight_n)
        push!(weights, weight_n)
        push!(pts, traces[min_l_ix])

        new_t, this_log_l = replace_fn(f, log_l_n, params...)
        traces[min_l_ix] = new_t
        loglikelihoods[min_l_ix] = this_log_l

        # estimate entropy 
        normed_weights = weights .- log(Z)
        if n > 1
            entropy = -1.0 * sum(exp.(normed_weights) .* normed_weights)
            breakpoint = Int(num_points * ceil(entropy))
            if n > breakpoint
                break
            end
        end
        n += 1
    end
    # approximation for the remainder of the sampled traces
    append!(pts, traces)
    x_last = exp(-(n - 1) / num_points)
    x_n = exp(-n / num_points)
    w_n = x_last - x_n
    for t in traces
        push!(weights, loglikelihood(t) + log(w_n))
    end
    nested_results(pts, weights)
end

@doc raw"""
    nested(f; params = (), num_points :: Int64 = 1)

Run nested sampling using internal rejection method.
"""
nested(f; params = (), num_points :: Int64 = 1) = nested(f, rejection; params = params, num_points = num_points)

export nested, sample