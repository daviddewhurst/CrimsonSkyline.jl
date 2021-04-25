function guide(t ::Trace, data)
    μ_loc = parameter(t, :μ_loc, 0.0)
    μ_log_scale = parameter(t, :μ_log_scale, 0.0)
    σ_loc = parameter(t, :σ_loc, 0.0)
    σ_log_scale = parameter(t, :σ_log_scale, 0.0)
    sample(t, :μ, Normal(μ_loc, exp(μ_log_scale)))
    sample(t, :σ, LogNormal(σ_loc, exp(σ_log_scale)))
end

function model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :μ, Normal(0.0, 1.0))
    scale = sample(t, :σ, Gamma(2.0, 2.0))
    for (i, d) in enumerate(data)
        observe(t, (:data, i), Normal(loc, scale), d)
    end
end

@testset "guide construction 1" begin
    data = [1.0, -2.0, 3.1]
    guide_trace = trace()
    guide(guide_trace, data)
    logprob!(guide_trace)
    model_trace, g = replay(model, guide_trace)
    g(data)
    logprob!(model_trace)
    mc_elbo = model_trace.logprob_sum - guide_trace.logprob_sum
    @info "MC elbo ≈ $mc_elbo"
end