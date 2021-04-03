using Test
using Logging
using Distributions: Normal, Poisson, Gamma, LogNormal, Bernoulli
using StatsBase: mean, std
using PrettyPrint: pprintln

using CrimsonSkyline

@testset "node construction" begin 
    dist = Normal()
    address = :z 
    n = node(Float64, address, dist, false, Nonstandard())
    @test n.logprob == 0.0
end

@testset "trace construction" begin
    dist = Normal()
    address = :z 
    n = node(Float64, address, dist, false, Nonstandard())
    t = trace()
    t[address] = n
    @test t[address] == n
end

@testset "sample construction" begin
    t = trace()
    s_d = sample(t, :count, Poisson(3.0))
    @info "Sampled $s_d"
    @info "Sample node is $(t[:count])"
    @test t[:count].value == s_d
end

function program!(t :: Trace, n :: Int)
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, Gamma(2.0, 2.0))
    log_rate = sample(t, :log_rate, Normal(loc, scale))
    data = sample(t, :data, Poisson(exp(log_rate)), (n,))
    data
end

@testset "tracing program" begin
    t = trace()
    n_datapoints = 10
    values = program!(t, n_datapoints)
    logprob!(t)
    @info "Joint density of execution = $(t.logprob_sum)"
    @test :loc in keys(t)
    @test :data in keys(t)
end

function program2!(t :: Trace, data)
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, Gamma(2.0, 2.0))
    log_rate = sample(t, :log_rate, Normal(loc, scale))
    obs = observe(t, :data, Poisson(exp(log_rate)), data)
    obs
end

@testset "observing data" begin
    data = [1, 1, 2, 1, 5]
    t = trace()
    obs = program2!(t, data)
    logprob!(t)
    @test obs == data
    # log likelihood is always >= log joint density
    log_joint = logprob(t)
    log_likelihood = loglikelihood(t)
    @info "log p(x, z) = $log_joint"
    @info "log p(x | z) = $log_likelihood"
    @test log_likelihood >= log_joint
    
    # copy the old trace and demonstrate modification
    t_copy = deepcopy(t)
    obs = program2!(t, data)
    @info ":loc node in copied original: $(t_copy[:loc])"
    @info ":loc node in modified original: $(t[:loc])"
    @test t[:loc].value != t_copy[:loc].value
end

@testset "replay" begin
    data = [1, 1, 2, 1, 5]
    # trace the program
    t = trace()
    obs = program2!(t, data)
    logprob!(t)
    # modify a trace element
    t2 = deepcopy(t)
    t2[:scale] = node(2.0, :scale, t[:scale].dist, false, Nonstandard())
    @test t[:scale].value != t2[:scale].value
    @test t[:scale].dist == t2[:scale].dist
    # replay the program using the new trace
    (r_obs, new_t) = replay(program2!, t2, data)
    logprob!(new_t)
    @test r_obs == data
    @test t[:scale].dist == new_t[:scale].dist
    @test t[:log_rate].dist != new_t[:log_rate].dist
    @test new_t[:scale].value == 2.0
    @info "Old scale node: $(t[:scale])"
    @info "New scale node: $(new_t[:scale])"
    @info "Old log_rate node: $(t[:log_rate])"
    @info "New log_rate node: $(new_t[:log_rate])"
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

function normal_graph_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, LogNormal())
    for i in 1:length(data)
        observe(t, (:obs, i), Normal(loc, scale), data[i]; pa=(:loc, :scale))
    end
end

@testset "get graph materials from trace" begin
    t = trace()
    data = [1.0, -4.1]
    r = normal_graph_model(t, data)
    @test all([(:obs, i) == t[:loc].ch[i].address for i in 1:length(data)])
    @test all([(:obs, i) == t[:scale].ch[i].address for i in 1:length(data)])
    @test :loc == t[(:obs, 1)].pa[1].address
    @test :scale == t[(:obs, 1)].pa[2].address
end

@testset "get straight graph from trace" begin
    t = trace()
    data = [1.0, -4.1]
    normal_graph_model(t, data)
    info, g = graph(t)
    @test length(info[:scale]["pa"]) == 0
    @test length(info[(:obs, 1)]["pa"]) == 2
    @test g[:scale] == [(:obs, 1), (:obs, 2)]
end

function test_switch_model(t :: Trace)
    z = sample(t, :z, Bernoulli(0.5))
    loc1 = sample(t, :loc1, Normal(0.0, 1.0))
    loc2 = sample(t, :loc2, Normal(1.0, 1.0))
    val = transform(t, :val,
        (z, a, b) -> if z < 1 a else b end,
        (z, loc1, loc2);
        pa = (:z, :loc1, :loc2)
    )
    data = sample(t, :data, Normal(val, 1.0); pa = (:val,))
    data
end

@testset "get transformed graph from trace" begin
    t = trace()
    test_switch_model(t)
    info, g = graph(t)
    @test info[:val]["interpretation"] == "Deterministic()"
    @test g[:val] == [:data]
    @test info[:val]["pa"] == [:z, :loc1, :loc2]
end