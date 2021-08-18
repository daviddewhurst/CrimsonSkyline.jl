function weird_model(t::Trace)
    choice = sample(t, "choice", Beta(1.0, 1.0))
    if choice > 0.5
        sample(t, "branch1", Normal())
    else
        sample(t, "branch2", LogNormal())
    end
end

@testset "table creation" begin 
    sqm = SQLModel(weird_model; num_iterations=500)
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

function regression(t::Trace, X::Vector{Float64}, y::Vector{Float64})
    alpha = sample(t, "alpha", Normal())
    beta = sample(t, "beta", Normal())
    sigma = sample(t, "sigma", LogNormal())
    mu = alpha .+ X .* beta
    observe(t, "obs", MvNormal(mu, sigma), y)
end

@testset "query posterior" begin
    N = 50
    X = randn(N,)
    true_alpha = 2.0
    true_beta = -0.3
    true_sigma = 0.25
    y = true_alpha .+ X .* true_beta .+ randn(N,) .* true_sigma
    train_N = 25
    sqm = SQLModel(
        regression;
        params = (X[1:train_N], y[1:train_N]),
        method = METROPOLIS,
        num_iterations = 21000,
        inference_params = Dict("burn" => 1000, "thin" => 100)
    )
    # examine posterior coefficients for "executive consumption"
    q = """
        SELECT 
            AVG(alpha) as alpha_mean,
            AVG(beta) as beta_mean,
            AVG(sigma) as noise_level
        FROM regression_results;
    """
    @info query(sqm, q)
end