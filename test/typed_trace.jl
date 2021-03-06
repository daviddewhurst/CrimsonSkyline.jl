function tt_normal_model_test(t::Trace, data::Vector{T}, params...) where T
    loc = sample(t, "loc", Normal())
    scale = sample(t, "scale", LogNormal())
    obs = Vector{Float64}(undef, length(data))
    for (i, d) in enumerate(data)
        obs[i] = observe(t, "obs $i", Normal(loc, scale), d)
    end
    obs
end

@testset "typed trace 1" begin
    data = rand(5)
    # to compile
    _ = likelihood_weighting(tt_normal_model_test, data)
    _ = likelihood_weighting(tt_normal_model_test, (String, Float64), data)
    # to time
    @time ut_res = likelihood_weighting(tt_normal_model_test, data; nsamples=100000)
    @time t_res = likelihood_weighting(tt_normal_model_test, (String, Float64), data; nsamples=100000)
end

function is_proposal(t, data, loc_guess, std_guess)
    propose(t, "loc", Normal(loc_guess, std_guess * 2.0))
    propose(t, "scale", truncated(Normal(std_guess, std_guess * 2.0), 0.1, Inf))
end

@testset "typed trace 2" begin
    data = rand(5)
    loc_guess = mean(data)
    std_guess = std(data)
    types = (String, Float64)
    _ = importance_sampling(tt_normal_model_test, is_proposal; params = (data, loc_guess, std_guess))
    _ = importance_sampling(tt_normal_model_test, is_proposal, types; params = (data, loc_guess, std_guess))
    @time importance_sampling(tt_normal_model_test, is_proposal; params = (data, loc_guess, std_guess), nsamples=100000)
    @time importance_sampling(tt_normal_model_test, is_proposal, types; params = (data, loc_guess, std_guess), nsamples=100000)
end

tt_loc_proposal(t0, t1, params...) = propose(t1, "loc", Normal(t0["loc"].value, 0.5))
tt_scale_proposal(t0, t1, params...) = propose(t1, "scale", LogNormal(log(t0["scale"].value), 0.5))

@testset "typed trace 3" begin
data = rand(5)
# to compile
proposals = [tt_loc_proposal, tt_scale_proposal]
types = (String, Float64)
_ = mh(tt_normal_model_test, proposals; params = (data,), inverse_verbosity=Inf)
_ = mh(tt_normal_model_test, proposals, types; params = (data,), inverse_verbosity=Inf)
# to time
@time ut_res = mh(tt_normal_model_test, proposals; params = (data,), burn=1000, thin=100, num_iterations=50000, inverse_verbosity=Inf)
@time t_res = mh(tt_normal_model_test, proposals, types; params = (data,), burn=1000, thin=100, num_iterations=50000, inverse_verbosity=Inf)
end