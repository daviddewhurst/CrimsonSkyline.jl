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
        (z, a, b) -> z < 1 ? a : b,
        (z, loc1, loc2);
        pa = (:z, :loc1, :loc2)
    )
    obs_scale = transform(t, :obs_scale, () -> 1.0, (); pa = ())
    sample(t, :data, Normal(val, obs_scale); pa = (:val, :obs_scale))
end

@testset "get transformed graph from trace" begin
    t = trace()
    test_switch_model(t)
    ir = graph_ir(t)
    @test ir.info[:val]["interpretation"] == DETERMINISTIC
    @test ir.graph[:val] == [:data]
    @test ir.info[:val]["pa"] == [:z, :loc1, :loc2]
end

function switch_model_2(t :: Trace, data)
    z = sample(t, :z, Bernoulli(0.5))
    loc1 = sample(t, :loc1, Normal(0.0, 1.0))
    loc2 = sample(t, :loc2, Normal(1.0, 1.0))
    val = transform(t, :val,
        (z, a, b) -> z < 1 ? a : b,
        (z, loc1, loc2);
        pa = (:z, :loc1, :loc2)
    )
    obs_scale = sample(t, :obs_scale, LogNormal())
    observe(t, :data, Normal(val, obs_scale), data; pa = (:val, :obs_scale))
end

@testset "sample graph 1" begin
    data = 2.0
    t = trace()
    switch_model_2(t, data)
    g = graph_ir(t)
    (s, lp) = sample(g)
    @info "Sampled from switch model graph: $s"
    @info "Log prob of sample: $lp"
end

@testset "get factor graph from graph ir" begin
    t = trace()
    test_switch_model(t)
    factor_graph = factor(t)
    # verify factor construction
    @test factor_graph.factor_to_node[0] == Set((:data, :val, :obs_scale))
    @test factor_graph.factor_to_node[1] == Set((:obs_scale,))
    @test factor_graph.factor_to_node[2] == Set((:val, :z, :loc1, :loc2))
    @test factor_graph.factor_to_node[3] == Set((:loc2,))
    @test factor_graph.factor_to_node[4] == Set((:loc1,))
    @test factor_graph.factor_to_node[5] == Set((:z,))
    @test factor_graph.node_to_factor[:z] == Set((5, 2))
    @test factor_graph.node_to_factor[:loc1] == Set((4, 2))
    @test factor_graph.node_to_factor[:loc2] == Set((3, 2))
    @test factor_graph.node_to_factor[:val] == Set((0, 2))
    @test factor_graph.node_to_factor[:obs_scale] == Set((0, 1))
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