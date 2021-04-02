using Test
using Logging
using Distributions: Normal, Poisson, Gamma
using PrettyPrint: pprintln

using CrimsonSkyline

@testset "node construction" begin 
    dist = Normal()
    address = :z 
    n = node(Float64, address, dist)
    @test n.logprob == 0.0
end

@testset "trace construction" begin
    dist = Normal()
    address = :z 
    n = node(Float64, address, dist)
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

@testset "tracing program" begin
    function program!(t :: Trace, n :: Int)
        loc = sample(t, :loc, Normal())
        scale = sample(t, :scale, Gamma(2.0, 2.0))
        log_rate = sample(t, :log_rate, Normal(loc, scale))
        data = sample(t, :data, Poisson(exp(log_rate)), n)
        data
    end

    t = trace()
    n_datapoints = 10
    values = program!(t, n_datapoints)
    logprob!(t)
    @info "Joint density of execution = $(t.logprob_sum)"
    pprintln(t)
end
