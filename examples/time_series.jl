import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Normal, Gamma, LogNormal
using Logging
using StatsBase: mean, std
using Random: seed!

seed!(2021)

function random_walk(t :: Trace, data :: Array{Float64, 2})
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, Gamma(3.0, 5.0))
    shape = size(data)
    for n in 1:shape[1]
        x_n = sample(t, (:ic, n), Normal())
        for t_ix in 1:shape[2]
            x_n = observe(t, (:data, n, t_ix), Normal(loc + x_n, scale), data[n, t_ix])
        end
    end
end

function global_trend(t :: Trace, data :: Array{Float64, 2})
    α = sample(t, :α, Normal())
    β = sample(t, :β, Normal())
    σ = sample(t, :σ, LogNormal())
    shape = size(data)
    for n in 1:shape[1]
        for t_ix in 1:shape[2]
            observe(t, (:data, n, t_ix), Normal(α + β * t_ix, σ), data[n, t_ix])
        end
    end
end

function main()
    n = 10
    t = 20

    @info "Random walk inference"
    loc = 1.0
    scale = 1.5
    noise = rand(n, t)
    data = cumsum(loc .+ scale .* noise, dims=2)
    @info "True loc = $loc, true scale = $scale"
    @time results = mh(random_walk; params=(data,), burn=1000, thin=50, num_iterations=50000)
    @info "Estimated loc = ($(mean(results, :loc)) ± $(2.0 * std(results, :loc)))"
    @info "Estimated scale = ($(mean(results, :scale)) ± $(2.0 * std(results, :scale)))"

    @info begin
        """
        Time series model comparison
        Comparing (prior) evidence for random walk model vs global trend
        """
    end
    nsamples = 5000
    rw_results = likelihood_weighting(random_walk, data; nsamples = nsamples)
    gt_results = likelihood_weighting(global_trend, data; nsamples = nsamples)
    log_evidence_rw = log_evidence(rw_results)
    log_evidence_gt = log_evidence(gt_results)
    @info begin 
        """
        log p(x | random walk) ≈ $log_evidence_rw
        log p(x | global trend) ≈ $log_evidence_gt
        """
    end
    @info "Comparing model quality using AIC"
    aic_rw = aic(rw_results)
    aic_gt = aic(gt_results)
    @info "AIC(random walk) ≈ $aic_rw, AIC(global trend) ≈ $aic_gt"

end

main()