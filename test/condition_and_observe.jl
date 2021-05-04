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