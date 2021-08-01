@testset "sf -> node" begin
    function normal_model(t, x)
        loc = sample(t, "loc", Normal())
        log_scale = sample(t, "log_scale", Normal())
        observe(t, "data", Normal(loc, exp(log_scale)), x)
    end

    sf = TypedSF(normal_model, Maybe{Float64}, Float64)
    input = 3.0
    csf = compile(sf, input, ["log_scale", "loc"])

    t = trace()
    sample(t, "test", csf)
    @info t
    @test abs(t["test"].value["loc"] - input) < 0.1
end