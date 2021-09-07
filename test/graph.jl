function to_graph_model(t::Trace)
    prob = sample(t, "prob", Beta(2.0, 2.0))
    num = sample(t, "num", truncated(Geometric(prob), 1, Inf); pa = ("prob",))
    sample(t, "count", Binomial(num, prob); pa = ("num", "prob"))
end

@testset "graph creation" begin
    g = pgm(to_graph_model)
    @info g
end