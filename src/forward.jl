struct Forward <: InferenceType end
const FORWARD = Forward()

forward_sampling_results() = NonparametricSamplingResults{Forward}(FORWARD,Array{Float64, 1}(), Array{Any, 1}(), Array{Trace, 1}())

@doc raw"""
    function forward_sampling(f; params = (), num_iterations = 1)

Draws samples from the model's joint density. Equivalent to calling
`f` in a loop `num_iterations` times, but results are collected in a 
`NonparametricSamplingResults` for easier postprocessing.
"""
function forward_sampling(f; params = (), num_iterations = 1)
    results = forward_sampling_results()
    for n in 1:num_iterations
        t = trace()
        r = f(t, params...)
        push!(results.return_values, r)
        push!(results.traces, t)
        push!(results.log_weights, logprob(t))
    end
    results
end

export Forward, FORWARD, forward_sampling