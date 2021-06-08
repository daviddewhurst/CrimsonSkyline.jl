function wider_normal_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal(0.0, 10.0))
    scale = sample(t, :scale, LogNormal())
    for i in 1:length(data)
        observe(t, (:obs, i), Normal(loc, scale), data[i])
    end
end

@testset "prior metropolis proposal 1" begin
    data = randn(100) .+ 4.0
    t = trace()
    wider_normal_model(t, data)
    @info "True loc = 4.0"
    @info "True std = 1.0"
    
    locs = []
    scales = []
    lls = []

    for i in 1:2500
        t = mh_step(t, wider_normal_model; params = (data,))
        push!(locs, t[:loc].value)
        push!(scales, t[:scale].value)
        push!(lls, loglikelihood(t))
    end

    @info "inferred E[loc] = $(mean(locs[1000:end]))"
    @info "inferred E[scale] = $(mean(scales[1000:end]))"
    @info "approximate p(x) = sum_z p(x|z) = $(mean(lls[1000:end]))"
end

@testset "prior metropolis proposal 2" begin
    data = randn(10) .+ 4.0
    @info "True loc = 4.0"
    @info "True std = 1.0"
    results = mh(wider_normal_model; params=(data,))
    mean_loc = mean(results, :loc)
    std_loc = std(results, :loc)
    @info "Mean loc = $mean_loc"
    @info "Std loc = $std_loc"
end

loc_proposal(old_t :: Trace, new_t :: Trace, data) = propose(new_t, :loc, Normal(old_t[:loc].value, 0.25))
scale_proposal(old_t :: Trace, new_t :: Trace, data) = propose(new_t, :scale, truncated(Normal(old_t[:scale].value, 0.25), 0.0, Inf))

@testset "general metropolis proposal 1" begin
    data = randn(100) .+ 4.0
    t = trace()
    wider_normal_model(t, data)

    locs = []
    scales = []
    lls = []

    for i in 1:2500
        t = mh_step(t, wider_normal_model, loc_proposal; params = (data,))
        t = mh_step(t, wider_normal_model, scale_proposal; params = (data,))
        push!(locs, t[:loc].value)
        push!(scales, t[:scale].value)
        push!(lls, loglikelihood(t))
    end

    @info "inferred E[loc] = $(mean(locs[1000:end]))"
    @info "inferred E[scale] = $(mean(scales[1000:end]))"
    @info "approximate p(x) = sum_z p(x|z) = $(mean(lls[1000:end]))"
end

@testset "generic metropolis proposal 2" begin
    data = randn(100) .+ 4.0
    @info "Using full results struct"
    results = mh(wider_normal_model, [loc_proposal, scale_proposal]; params = (data,), inverse_verbosity = Inf)
    @info "Using bare results struct"
    results = mh(wider_normal_model, [loc_proposal, scale_proposal], (:loc,); params = (data,), inverse_verbosity = Inf)
end

function random_sum_model(t :: Trace, data)
    n = sample(t, :n, Geometric(0.1))
    loc = 0.0
    for i in 1:(n + 1)
        loc += sample(t, (:loc, i), Normal())
    end
    obs = Array{Float64, 1}()
    for j in 1:length(data)
        o = observe(t, (:data, j), Normal(loc, 1.0), data[j])
        push!(obs, o)
    end
    obs
end

function random_n_proposal(old_trace, new_trace, params...)
    old_n = float(old_trace[:n].value)
    if old_n > 0
        propose(new_trace, :n, Poisson(old_n))
    else
        propose(new_trace, :n, Poisson(1.0))
    end
end

function gen_loc_proposal(old_trace, new_trace, ix, params...)
    propose(new_trace, (:loc, ix), Normal(old_trace[(:loc, ix)].value, 0.25))
end

@testset "generic metropolis proposal 3" begin
    t = trace()
    data = random_sum_model(t, fill(nothing, 25))
    @info "True :n = $(t[:n].value)"
    random_sum_model(t, data)
    
    ns = []
    loc_proposals = [
        (o, n, params...) -> gen_loc_proposal(o, n, i, params...) for i in 1:100
    ]

    for i in 1:5000
        t = mh_step(t, random_sum_model, random_n_proposal; params=(data,))
        for j in 1:(t[:n].value + 1)
            t = mh_step(t, random_sum_model, loc_proposals[j]; params=(data,))
        end
        push!(ns, t[:n].value)
    end

    @info "Posterior E[:n] = $(mean(ns[1000:end]))"
end

const Maybe{T} = Union{T, Nothing}

function eye(d :: Int)
    mat = zeros(Float64, d, d)
    for i in 1:d
        mat[i, i] = 1.0
    end
    mat
end

function linear_model!(t :: Trace, out :: Vector{Float64}, X :: Matrix{Float64}, y :: Maybe{Vector{Float64}})
    size_X = size(X)
    D = size_X[1]
    N = size_X[2]
    β = sample(t, :β, MvNormal(zeros(D), eye(D)))
    scale = sample(t, :scale, LogNormal(0.0, 1.0))
    loc = X' * β
    if y === nothing
        for n in 1:N
            out[n] = sample(t, (:y, n), Normal(loc[n], scale))
        end
    else
        for n in 1:N
            out[n] = observe(t, (:y, n), Normal(loc[n], scale), y[n])
        end
    end
end

beta_proposal(old_t :: Trace, new_t :: Trace, _, _, _) = propose(new_t, :β, MvNormal(old_t[:β].value, 0.5 .* eye(length(old_t[:β].value))))
scale_proposal(old_t :: Trace, new_t :: Trace, _, _, _) = propose(new_t, :scale, truncated(Normal(old_t[:scale].value, 0.25), 0.0, Inf))

@testset "generic metropolis proposal 4" begin
    D = 10
    N = 200
    X = randn(D, N)
    out = Vector{Float64}(undef, N)
    @info "Using model to generate dataset"
    t = trace()
    linear_model!(t, out, X, nothing)
    y = [t[(:y, n)].value for n in 1:N]
    β_true = t[:β].value
    @info "True regression coefficients: $β_true"

    @info "Conducting inference using user-defined Metropolis kernels"
    # smoketest only --- low number of iterations. see examples/basic.jl for actual inference results
    @time results = mh(
        linear_model!,
        [beta_proposal, scale_proposal];
        params = (out, X, y),
        burn = 50,
        thin = 10,
        num_iterations = 500
    )
    β_mean = mean(results, :β)
    β_std = std(results, :β)
    @info "Posterior mean regression coefficients = $β_mean"
    @info "Posterior std regression coefficients = $β_std"
end