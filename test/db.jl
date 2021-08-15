function weird_model(t::Trace)
    choice = sample(t, "choice", Beta(1.0, 1.0))
    if choice > 0.5
        sample(t, "branch1", Normal())
    else
        sample(t, "branch2", LogNormal())
    end
end

@testset "table creation" begin 
    sqm = SQLiteModel(weird_model; num_iterations=100)
    q = """
        SELECT 
            AVG(branch1_logprob) as avg_1_logprob,
            AVG(branch2_logprob) as avg_2_logprob
        FROM 
            weird_model_results
        WHERE 
            choice < 0.25;
    """
    result = query(sqm, q)
    @info result
end