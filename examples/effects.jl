import Pkg
Pkg.activate("..")

using Distributions: Normal, Geometric, Poisson, truncated
using DataStructures
using CrimsonSkyline
using Random: seed!
using Logging
using StatsBase: mean, std
using PrettyPrint: pprintln

seed!(2021)

function normal_model(t::Trace, data::Vector{T}) where T
    loc = sample(t, "loc", Normal())
    log_scale = sample(t, "log_scale", Normal())
    obs = Vector{Float64}(undef, length(data))
    for (i, d) in enumerate(data)
        obs[i] = observe(t, "obs $i", Normal(loc, exp(log_scale)), d)
    end
    (t, obs)
end

function augmented_normal_model(t::Trace, data1::Vector{T1}, data2::Vector{T2}) where {T1, T2}
    normal_model(t, data1)
    noisy_z_sq = sample(t, "noisy z squared", Normal(t["obs 1"].value ^ 2, exp(t["log_scale"].value)))
    for (i, d) in enumerate(data2)
        observe(t, "another obs $i", Normal(noisy_z_sq, exp(t["log_scale"].value)), d)
    end
    t
end

function demo_replay()
    @warn "\n~~~Demo replay effect~~~\n"
    # replay makes a stochastic function behave as though it had 
    # sampled values from a different passed trace
    data = randn(1)
    (t, obs) = normal_model(trace(), data)
    @info "First sampled trace: $t"
    t_new, replayed_normal_model = replay(normal_model, t)
    @info "Replayed trace *before* replayed model is called: $t_new"
    replayed_normal_model(data)
    @info "Replayed trace *after* replayed model is called: $t_new"
    # replay can be used to restrict only some elements of traces 
    # this can be accomplished by model composition, as below, or through
    # use of a block effect, demonstrated in demo_block
    # here, we replay only the normal_model portion of the trace through
    # augmented_normal_model
    data2 = [3.0, 4.0]
    t_new, replayed_augmented = replay(augmented_normal_model, t)
    @info "Replaying trace through augmented model: replayed trace *before* calling replayed model: $t_new"
    replayed_augmented(data, data2)
    @info "Replayed trace *after* replayed augmented model is called: $t_new"
end

function demo_block()
    @warn "\n~~~Demo block effect~~~\n"
    # block hides sample sites from the outside world, i.e., converts them 
    # into untraced randomness. While they still affect internal model 
    # dynamics, their values are not recorded in the trace (e.g., they do not
    # contribute to model density). Because of this, block should be used 
    # with caution 
    data = randn(2)
    (t, _) = normal_model(trace(), data)
    # for example, we can convert log_scale into untraced_randomness
    (t, blocked_model) = block(normal_model, t, ("log_scale",))
    @info "Trace before blocked model is called: $t"
    blocked_model(data)
    @info "Trace after blocked model is called: $t"

    # replaying a trace through a blocked model
    @info "Replaying a trace through a blocked model "
    data2 = [3.0, 4.0]
    t = augmented_normal_model(t, data, data2)
    @info "First trace through augmented normal model: $t"
    blocked = block(augmented_normal_model, ("log_scale",))
    (t, replayed) = replay(blocked, t)
    replayed(data, data2)
    @info "New trace: $t"

    # replaying all but a single changed site -- this technique could be used
    # to create nuanced single site proposal kernels
    @info "Replaying all but a single changed site"
    t = augmented_normal_model(t, data, data2) 
    (t, blocked) = block(augmented_normal_model, t, ("loc",))
    @info "Trace before blocked model execution: $t"
    blocked(data, data2)
    @info "Trace after blocked model execution: $t"
    (t, replayed) = replay(augmented_normal_model, t)
    @info "Trace before replayed model execution: $t"
    replayed(data, data2)
    @info "Trace after replayed model execution: $t"

    # watch out for tricky bugs -- augmented_normal_model depends on the *trace* value
    # (not return value) of sampling log_scale, so blocking this site could have
    # undesirable effects...
    @warn "Block could interfere with downstream sites depending on trace values!"
    t = augmented_normal_model(t, data, data2) 
    (t, blocked) = block(augmented_normal_model, t, ("log_scale",))
    try
        blocked(data, data2)
    catch e
        if e isa KeyError
            @warn "Error when running blocked model: $e"
        end
    end
end

# use this version to demonstrate the updating process

function normal_model(t::Trace, data::Vector{T}, verbose::Bool) where T
    loc = sample(t, "loc", Normal(0.0, 4.0))
    log_scale = sample(t, "log_scale", Normal())
    if verbose
        @info "Normal model: loc = $loc, log_scale = $log_scale"
    end
    obs = Vector{Float64}(undef, length(data))
    for (i, d) in enumerate(data)
        obs[i] = observe(t, "obs $i", Normal(loc, exp(log_scale)), d)
    end
    if verbose
        @info "Normal model: obs = $obs"
    end
    (t, obs)
end

function demo_update()
    @warn "\n~~~Demo update effect~~~\n"
    # grab some samples from the prior
    data = randn(2) .+ 2.0
    sites = ["loc", "log_scale"]
    prior_samples = forward_sampling(normal_model; params = (data, false), num_iterations = 100)
    for site in sites
        @info "Prior mean $site: $(mean(prior_samples[site]))"
        @info "Prior std $site: $(std(prior_samples[site]))"
    end
    # now, perform inference
    inference_results = mh(normal_model; params = (data, false), burn=500, thin=50, num_iterations = 5000)
    for site in sites
        @info "Posterior mean $site: $(mean(inference_results, site))"
        @info "Posterior std $site: $(std(inference_results, site))"
    end
    # update from prior to empirical posterior
    posterior_normal_model = update(normal_model, inference_results)
    # draw some values from the posterior predictive
    n_pp = 10
    map((n) -> posterior_normal_model(trace(), [nothing], true)[end][end], 1:n_pp)
end

function demo_condition()
    @warn "\n~~~Demo condition effect~~~\n"
    # condition imposes a "soft constraint" when applying evidence to a node
    # this means that the model isn't guaranteed to sample the conditioned value at 
    # that node, but when it does not sample the conditioned value, the log prob of
    # the trace is = -Inf. Practically, condition *does* enforce a hard constraint on 
    # node value for all models with static structure (e.g., see forecast.jl for another 
    # applied example). However, in models with open universe structure, applying evidence 
    # to a node that only exists conditioned on the values of upstream choices may result 
    # in that node not being reached for a particular model evaluation. 
    # This results in the log prob of the trace = -Inf.
    data = randn(2) .+ 2.0
    @info "Before conditioning"
    (t, _) = normal_model(trace(), data, true)
    @info "Unconditioned trace: $t"
    # condition loc to be equal to 3.0
    loc = 3.0
    @info "Condition loc = $loc"
    @info "After conditioning"
    conditioned_normal_model = condition(normal_model, Dict("loc" => loc))
    (t, (t, r)) = conditioned_normal_model(trace(), data, true)
    @info "Conditioned trace: $t"
end

function main()
    demo_replay()
    demo_block()
    demo_update()
    demo_condition()
end 

main()