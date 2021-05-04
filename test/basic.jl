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