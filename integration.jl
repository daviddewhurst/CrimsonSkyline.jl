using Pkg
Pkg.activate(".")
using Distributions: Bernoulli, Gamma, Normal, LogNormal
using Logging
using StatsBase: mean, std

using CrimsonSkyline

### some interesting models ###

function switch_model(t :: Trace)
    z = sample(t, :z, Bernoulli(0.5))
    loc1 = sample(t, :loc1, Normal(0.0, 1.0))
    loc2 = sample(t, :loc2, Normal(1.0, 1.0))
    val = transform(t, :val,
        (z, a, b) -> if z < 1 a else b end,
        (z, loc1, loc2);
        pa = (:z, :loc1, :loc2)
    )
    data = observe(t, :data, Normal(val, 1.0); pa = (:val,))
    data
end

function normal_model(t :: Trace, data :: D) where D <: AbstractArray
    shared_scale = sample(t, :shared_scale, LogNormal(-1.0, 0.5))
    loc = sample(t, :loc, Normal(0.0, shared_scale))
    scale = sample(t, :scale, LogNormal(0.0, shared_scale))
    out = Array{Float64, 1}()
    for i in 1:length(data)
        o = observe(t, (:obs, i), Normal(loc, scale), data[i])
        push!(out, o)
    end
    out
end

### integration testing -- importance sampling ###

function itest_normal_model()
    # generate data from the model
    @info "Likelihood weighting on shared scale normal model"
    t = trace()
    data = normal_model(t, fill(nothing, (25,)))
    @info "Generated $data from normal_model"
    true_ss = t[:shared_scale].value
    true_loc = t[:loc].value
    true_scale = t[:scale].value
    @info "Used shared scale = $true_ss"
    @info "Used loc = $true_loc"
    @info "Used scale = $true_scale"
    
    posterior = likelihood_weighting(normal_model, data; nsamples = 1000)
    log_p_x = log_evidence(posterior)
    @info "Log evidence = $log_p_x"

    post_loc = sample(posterior, :loc, 2000)
    post_scale = sample(posterior, :scale, 2000)
    post_shared_scale = sample(posterior, :shared_scale, 2000)
    @info "Posterior E[shared scale] = $(mean(post_shared_scale))"
    @info "Posterior E[loc] = $(mean(post_loc))"
    @info "Posterior E[scale] = $(mean(post_scale))"
    @info "Posterior Std[shared scale] = $(std(post_shared_scale))"
    @info "Posterior Std[loc] = $(std(post_loc))"
    @info "Posterior Std[scale] = $(std(post_scale))"
end


function main()
    itest_normal_model()
end

main()