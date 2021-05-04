function lomax_model(t :: Trace)
    scale = sample(t, :scale, Lomax(1.0, 2.0))
    observe(t, :data, Normal(0.0, scale), nothing)
end

@testset "created distributions: lomax" begin
    t = trace()
    lomax_model(t)
    @test typeof(t[:scale]) == Node{Symbol,Lomax{Float64},Float64,Float64}
end

function normal_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, LogNormal())
    for i in 1:length(data)
        observe(t, (:obs, i), Normal(loc, scale), data[i])
    end
end

@testset "likelihood weighting 1" begin
    true_loc = 2.0
    true_scale = 1.0
    @info "True loc = $true_loc"
    @info "True scale = $true_scale"
    n_dpts = 10
    data = true_loc .+ true_scale .* randn(n_dpts)
    prior_samples = prior(normal_model, (:loc, :scale), data; nsamples = 100)
    @info "Prior E[loc] = $(mean(prior_samples[:loc]))"
    @info "Prior E[scale] = $(mean(prior_samples[:scale]))"

    posterior = likelihood_weighting(normal_model, data; nsamples = 1000)
    log_p_x = log_evidence(posterior)
    @info "Log evidence = $log_p_x"

    post_loc = sample(posterior, :loc, 2000)
    post_scale = sample(posterior, :scale, 2000)
    @info "Posterior E[loc] = $(mean(post_loc))"
    @info "Posterior E[scale] = $(mean(post_scale))"
end

function is_proposal(t :: Trace, data :: Vector{Float64})
    loc_guess = mean(data)
    std_guess = std(data)
    propose(t, :loc, Normal(loc_guess, std_guess * 0.25))
    propose(t, :scale, truncated(Normal(std_guess, std_guess * 0.25), 0.1, Inf))
end

@testset "importance sampling 1" begin
    true_loc = 2.0
    true_scale = 1.0
    @info "True loc = $true_loc"
    @info "True scale = $true_scale"
    n_dpts = 10
    data = true_loc .+ true_scale .* randn(n_dpts)
    prior_samples = prior(normal_model, (:loc, :scale), data; nsamples = 100)
    @info "Prior E[loc] = $(mean(prior_samples[:loc]))"
    @info "Prior E[scale] = $(mean(prior_samples[:scale]))"

    posterior = importance_sampling(normal_model, is_proposal; params = (data,), nsamples = 1000)
    log_p_x = log_evidence(posterior)
    @info "Log evidence = $log_p_x"

    post_loc = sample(posterior, :loc, 2000)
    post_scale = sample(posterior, :scale, 2000)
    @info "Posterior E[loc] = $(mean(post_loc))"
    @info "Posterior E[scale] = $(mean(post_scale))"
end
