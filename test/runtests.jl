using Test
using Logging
using Distributions: Normal, Poisson, Gamma, LogNormal, Bernoulli, Geometric, truncated
using StatsBase: mean, std
using PrettyPrint: pprintln
using Random

using CrimsonSkyline

Random.seed!(2021)

@testset "node construction" begin 
    dist = Normal()
    address = :z 
    n = node(Float64, address, dist, false, NONSTANDARD)
    @test n.logprob == 0.0
end

@testset "trace construction" begin
    dist = Normal()
    address = :z 
    n = node(Float64, address, dist, false, NONSTANDARD)
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
    for i in 1:n
        sample(t, (:data, i), Poisson(exp(log_rate)))
    end
end

@testset "tracing program" begin
    t = trace()
    n_datapoints = 10
    program!(t, n_datapoints)
    logprob!(t)
    @info "Joint density of execution = $(t.logprob_sum)"
    @test :loc in keys(t)
    @test (:data, 3) in keys(t)
end

@testset "condition 1" begin
    t = trace()
    n_datapoints = 2
    program!(t, n_datapoints)
    @test t[(:data, 1)].interpretation == NONSTANDARD
    cp = condition(program!, Dict((:data, 1) => 4))
    t, _ = cp(t, n_datapoints)
    @test t[(:data, 1)].interpretation == STANDARD
    @test t[(:data, 1)].value == 4
    @info "(:data, 1) = $(t[(:data, 1)])"
end

function program2!(t :: Trace, data)
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, Gamma(2.0, 2.0))
    log_rate = sample(t, :log_rate, Normal(loc, scale))
    obs = Array{Int64, 1}(undef, length(data))
    for (i, d) in enumerate(data)
        obs[i] = observe(t, (:data, i), Poisson(exp(log_rate)), d)
    end
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
    t2[:scale] = node(2.0, :scale, t[:scale].dist, false, NONSTANDARD)
    @test t[:scale].value != t2[:scale].value
    @test t[:scale].dist == t2[:scale].dist
    # replay the program using the new trace
    t_new, replayed_f = replay(program2!, t2)
    r_obs = replayed_f(data)
    logprob!(t_new)
    @test r_obs == data
    @test t[:scale].dist == t_new[:scale].dist
    @test t[:log_rate].dist != t_new[:log_rate].dist
    @test t_new[:scale].value == 2.0
    @info "Old scale node: $(t[:scale])"
    @info "New scale node: $(t_new[:scale])"
    @info "Old log_rate node: $(t[:log_rate])"
    @info "New log_rate node: $(t_new[:log_rate])"
end

@testset "block" begin
    data = [1, 1, 2, 1, 5]
    t = trace()
    obs = program2!(t, data)
    logprob!(t)
    @test :loc in keys(t)

    t_new, blocked_f = block(program2!, t, (:loc,))
    r_obs = blocked_f(data)
    logprob!(t_new)
    @test !(:loc in keys(t_new))
end

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
    ir = graph_ir(t)
    @test length(ir.info[:scale]["pa"]) == 0
    @test length(ir.info[(:obs, 1)]["pa"]) == 2
    @test ir.graph[:scale] == [(:obs, 1), (:obs, 2)]
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
    ir = graph_ir(t)
    @test ir.info[:val]["interpretation"] == DETERMINISTIC
    @test ir.graph[:val] == [:data]
    @test ir.info[:val]["pa"] == [:z, :loc1, :loc2]
end

@testset "get factor graph from graph ir" begin
    t = trace()
    test_switch_model(t)
    factor_graph = factor(t)
    # verify factor construction
    @test factor_graph.factor_to_node[0] == Set((:data, :val))
    @test factor_graph.factor_to_node[1] == Set((:val, :z, :loc1, :loc2))
    @test factor_graph.factor_to_node[2] == Set((:loc2,))
    @test factor_graph.factor_to_node[3] == Set((:loc1,))
    @test factor_graph.factor_to_node[4] == Set((:z,))
    @test factor_graph.node_to_factor[:z] == Set((4, 1))
    @test factor_graph.node_to_factor[:loc1] == Set((3, 1))
    @test factor_graph.node_to_factor[:loc2] == Set((2, 1))
    @test factor_graph.node_to_factor[:val] == Set((1, 0))
    @test factor_graph.node_to_factor[:data] == Set((0,))
end

@testset "generate cpt" begin
    dims = Dict("cost" => ["high", "low"], "revenue" => ["high", "medium", "low"])
    c = cpt(dims)
    c[("high", "low")] = 0.4
    c[("high", "high")] = 0.2
    c[("low", "low")] = 0.3
    renormalize!(c)
    high_med = c[("high", "medium")]
    low_low = c[("low", "low")]
    high_low = c[("high", "low")]
    @info "P(cost=high, revenue=medium) = $high_med"
    @info "P(cost=low, revenue=low) = $low_low"
    @info "P(cost=high, revenue=low) = $high_low"
end

function wider_normal_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal(0.0, 10.0))
    scale = sample(t, :scale, LogNormal())
    for i in 1:length(data)
        observe(t, (:obs, i), Normal(loc, scale), data[i])
    end
end

@testset "prior metropolis proposal 1" begin
    data = randn(100) .+ 4.0
    t = trace()
    wider_normal_model(t, data)
    @info "True loc = 4.0"
    @info "True std = 1.0"
    
    locs = []
    scales = []
    lls = []

    for i in 1:2500
        t = mh_step(t, wider_normal_model; params = (data,))
        push!(locs, t[:loc].value)
        push!(scales, t[:scale].value)
        push!(lls, loglikelihood(t))
    end

    @info "inferred E[loc] = $(mean(locs[1000:end]))"
    @info "inferred E[scale] = $(mean(scales[1000:end]))"
    @info "approximate p(x) = sum_z p(x|z) = $(mean(lls[1000:end]))"
end

@testset "prior metropolis proposal 2" begin
    data = randn(10) .+ 4.0
    @info "True loc = 4.0"
    @info "True std = 1.0"
    results = mh(wider_normal_model; params=(data,))
    mean_loc = mean(results, :loc)
    std_loc = std(results, :loc)
    @info "Mean loc = $mean_loc"
    @info "Std loc = $std_loc"
end

loc_proposal(old_t :: Trace, new_t :: Trace, data) = propose(new_t, :loc, Normal(old_t[:loc].value, 0.25))
scale_proposal(old_t :: Trace, new_t :: Trace, data) = propose(new_t, :scale, truncated(Normal(old_t[:scale].value, 0.25), 0.0, Inf))


@testset "general metropolis proposal 1" begin
    data = randn(100) .+ 4.0
    t = trace()
    wider_normal_model(t, data)

    locs = []
    scales = []
    lls = []

    for i in 1:2500
        t = mh_step(t, wider_normal_model, loc_proposal; params = (data,))
        t = mh_step(t, wider_normal_model, scale_proposal; params = (data,))
        push!(locs, t[:loc].value)
        push!(scales, t[:scale].value)
        push!(lls, loglikelihood(t))
    end

    @info "inferred E[loc] = $(mean(locs[1000:end]))"
    @info "inferred E[scale] = $(mean(scales[1000:end]))"
    @info "approximate p(x) = sum_z p(x|z) = $(mean(lls[1000:end]))"
end

function random_sum_model(t :: Trace, data)
    n = sample(t, :n, Geometric(0.1))
    loc = 0.0
    for i in 1:(n + 1)
        loc += sample(t, (:loc, i), Normal())
    end
    obs = Array{Float64, 1}()
    for j in 1:length(data)
        o = observe(t, (:data, j), Normal(loc, 1.0), data[j])
        push!(obs, o)
    end
    obs
end

function random_n_proposal(old_trace, new_trace, params...)
    old_n = float(old_trace[:n].value)
    if old_n > 0
        propose(new_trace, :n, Poisson(old_n))
    else
        propose(new_trace, :n, Poisson(1.0))
    end
end

function gen_loc_proposal(old_trace, new_trace, ix, params...)
    propose(new_trace, (:loc, ix), Normal(old_trace[(:loc, ix)].value, 0.25))
end

@testset "generate metropolis proposal 2" begin
    t = trace()
    data = random_sum_model(t, fill(nothing, 25))
    @info "True :n = $(t[:n].value)"
    random_sum_model(t, data)
    
    ns = []
    loc_proposals = [
        (o, n, params...) -> gen_loc_proposal(o, n, i, params...) for i in 1:100
    ]

    for i in 1:5000
        t = mh_step(t, random_sum_model, random_n_proposal; params=(data,))
        for j in 1:(t[:n].value + 1)
            t = mh_step(t, random_sum_model, loc_proposals[j]; params=(data,))
        end
        push!(ns, t[:n].value)
    end

    @info "Posterior E[:n] = $(mean(ns[1000:end]))"
end