signatures(f) = map(m -> m.sig, methods(f).ms)
is_sf(f) = any(map(m -> Trace in m.parameters, signatures(f)))
check_sf(f) = !is_sf(f) && error("$f does not appear to be a stochastic function. (Try type annotation if it is.)")
export is_sf, check_sf

@doc raw"""
    function inference(f, method::Forward; params = (), inference_params = Dict())

No additional arguments.
"""
function inference(f, method::Forward; params = (), inference_params = Dict())
    forward_sampling(f; params = params, num_iterations = get(inference_params, "num_iterations", 1))
end

@doc raw"""
    function inference(f, method::LikelihoodWeighting; params = (), inference_params = Dict())

No additional arguments.
"""
function inference(f, method::LikelihoodWeighting; params = (), inference_params = Dict())
    likelihood_weighting(f, params...; nsamples = get(inference_params, "num_iterations", 1))
end

@doc raw"""
    inference(f; params = (), inference_params = Dict())

Do inference on the stochastic function `f`, passing as arguments to the inference algorithm 
the tuple of `params` and any additional arguments to the inference algorithm in `inference_params`.
To specify a particular inference algorithm pass an `InferenceType` as the second argument
to the function, i.e., 
`inference(f, i::InferenceType; params = (), inference_params = Dict())`.
All methods accept `inference_params["num_iterations"]::Int64` as the number of iterations of
inference to perform; the interpretation of that is inference algorithm-dependent.
**This default method performs inference using `LikelihoodWeighting()`.**
"""
inference(f; params = (), inference_params = Dict()) = inference_params(f, LW; params = params, inference_params = inference_params)

@doc raw"""
    function inference(f, method::ImportanceSampling; params = (), inference_params = Dict())

Additional arguments: `inference_params["kernel"] = q` 
where `q` is a proposal kernel for importance sampling
"""
function inference(f, method::ImportanceSampling; params = (), inference_params = Dict())
    !("kernel" in keys(inference_params)) && error("Must pass importance sampling kernel in inference_params")
    kernel = inference_params["kernel"]
    check_sf(kernel)
    importance_sampling(f, kernel; params = params, nsamples = get(inference_params, "num_iterations", 1))
end

@doc raw"""
    function inference(f, method::Nested; params = (), inference_params = Dict())

No additional arguments.
"""
function inference(f, method::Nested; params = (), inference_params = Dict())
    nested(f, rejection; params = params, num_points = get(inference_params, "num_iterations", 2))
end

@doc raw"""
    function inference(f, method::Metropolis; params = (), inference_params = Dict())

Additional arguments: `inference_params["kernels"]` 
which is a vector of >= 1 proposal kernels. If this is not passed, this method uses 
ancestor MH sampling.
"""
function inference(f, method::Metropolis; params = (), inference_params = Dict())
    with_kernels = "kernels" in keys(inference_params)
    if with_kernels
        kernels = inference_params["kernels"]
        map(k -> check_sf(k), kernels)
        mh(
            f, kernels;
            params = params,
            burn = get(inference_params, "burn", 0),
            thin = get(inference_params, "thin", 1),
            num_iterations = get(inference_params, "num_iterations", 2)
        )
    else
        @info "No kernel(s) passed, using ancestor sampling Metropolis."
        mh(
            f;
            params = params,
            burn = get(inference_params, "burn", 0),
            thin = get(inference_params, "thin", 1),
            num_iterations = get(inference_params, "num_iterations", 2)
        )
    end
end

export inference