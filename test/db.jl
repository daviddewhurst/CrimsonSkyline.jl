function weird_model(t::Trace)
    choice = sample(t, "choice", Beta(1.0, 1.0))
    if choice > 0.5
        sample(t, "branch1", Normal())
    else
        sample(t, "branch2", LogNormal())
    end
end

@testset "table creation" begin 
    db = table(weird_model; num_iterations=10)
    @info db
end