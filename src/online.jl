@doc raw"""
    function reflate!(r::ParametricSamplingResults{I}, mapping::Dict) where I<:InferenceType

Reflates the scale parameters of the `distributions` of `r`. The inferred scale parameters 
approximate the true uncertainty of the posterior distribution under the critical assumption
that the data generating process (DGP) remains constant. When the DGP is nonstationary,
it is necessary to intervene and increase the value of the scale parameters to properly 
account for uncertainty. `mapping` is a dict of `address => value` where `value` is either
a float or a positive definite matrix depending on if the distribution associated with `address`
is scalar- or vector-valued. 
"""
function reflate!(r::ParametricSamplingResults, mapping::Dict)
    for (a, value) in mapping
        d = r.distributions[a]
        p = params(d)
        r.distributions[a] = typeof(d)(p[1], value)
    end
end

function combine(v, pv::Vector{Float64})
    all(i > 0 for i in pv) || error("each element of pv must be >= 0")
    # cannot assume each results has the same set of addresses
    a_set = Set()
    for results in v
        union!(a_set, addresses(results))
    end
    a2mixture = Dict()
    for address in a_set
        weights = Float64[]
        dists = []
        for (i, results) in enumerate(v)
            if address in keys(results.distributions)
                push!(weights, i)
                push!(dists, results.distributions[address])
            end
        end
        if length(dists) == 0
            continue
        else
            weights ./= sum(weights)
            dists = convert(Vector{typeof(dists[1])}, dists)
            a2mixture[address] = MixtureModel(dists, weights)
        end
    end
    a2mixture
end

export reflate!, combine