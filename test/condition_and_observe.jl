@testset "condition 1" begin
    t = trace()
    n_datapoints = 2
    program!(t, n_datapoints)
    @test t[(:data, 1)].interpretation == NONSTANDARD
    cp = condition(program!, Dict((:data, 1) => 4))
    t, _ = cp(t, n_datapoints)
    @test t[(:data, 1)].interpretation == CONDITIONED
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

function plate_program!(t :: Trace, data)
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, Gamma(2.0, 2.0))
    log_rate = sample(t, :log_rate, Normal(loc, scale))
    plate(t, observe, :data, Poisson(exp(log_rate)), data)
end

@testset "plate observe" begin
    data = [1, 2, 3]
    t = trace()
    plate_program!(t, data)
    @info "Trace: $t"
    @test typeof(t[:data].value) == Vector{Int64}
    d = Poisson(exp(t[:log_rate].value))
    @test isapprox(t[:data].logprob_sum, sum(logpdf(d, i) for i in data))
end

@testset "plate latent" begin
    t = trace()
    plate(t, sample, "address", Normal(), 10)
    @info "Trace: $t"
    @test isapprox(t["address"].logprob_sum, sum(logpdf(Normal(), r) for r in t["address"].value))
end

@testset "plate mh inference" begin
    data = [1, 3, 1, 5, 4]
    results = mh(plate_program!; params = (data,))
    @info "Single results trace: $(results.traces[end])"
    @test typeof(results.traces[end][:data].value) == Vector{Int64}
end

@testset "plate replay" begin
    t = trace()
    f = (t,) -> plate(t, sample, "address", Normal(), 3)
    plate(t, sample, "address", Normal(), 3)
    @info "Original trace: $t"
    (new_t, g) = replay(f, t)
    @info "Replayed trace before function exec: $new_t"
    g()
    @info "Replayed trace after function exec: $new_t"
end

function expanded_plate_model(t, data)
    hl = sample(t, "hl", Normal())
    locs = plate(t, sample, "locs", Normal(hl, 1.0), length(data))
    plate(t, observe, "data", Normal, data, (locs, ones(length(data))))
end

@testset "expanded obs plate 1" begin
    t = trace()
    data = randn(3)
    expanded_plate_model(t, data)
    @info t
end

locs_proposal(t0, t1, params...) = propose(t1, "locs", MvNormal(t0["locs"].value, 1.0))
hl_proposal(t0, t1, params...) = propose(t1, "hl", Normal(t0["hl"].value, 1.0))

@testset "expanded obs plate 2" begin
    data = randn(10)
    results = mh(expanded_plate_model, [locs_proposal, hl_proposal]; params = (data,))
end