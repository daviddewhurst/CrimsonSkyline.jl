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

@doc raw"""
    function combine(v, pv::Vector{Float64})

Combines iterable of `ParametricSamplingResults`, `v`, into a single `ParametricSamplingResults`
with mixture distribution parametric posteriors. `pv` is a weight vector (all elements > 0)
that assigns relative weight to the mixture components in each `ParametricSamplingResults`.
Mathematically, for each address ``a``, the resulting mixture distribution is

```math
p_a(z) = \sum_{r \in \mathtt{v}:\ a \in \mathcal{A}(r)} p^r_a(z)\  w^a_r,
```

where each ``p^r_a(z)`` is the ``r``-th `ParametricSamplingResults`'s estimated
parametric posterior marginal for address ``a``, if it exists, and 
``w^a_r = \mathtt{pv}_r \Big/ \left( \sum_{r \in \mathtt{v}:\ a \in \mathcal{A}(r)} \mathtt{pv}_r\right)``.
"""
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
                push!(weights, pv[i])
                push!(dists, results.distributions[address])
            end
        end
        if length(dists) == 0
            continue
        else
            weights ./= sum(weights)
            # assumes all dists are of same type for given addreess, currently this is how 
            # ParametricSamplingResults works but warning that this could change
            # if that assumption is incorrect this will fail on type conversion
            dists = convert(Vector{typeof(dists[1])}, dists)
            a2mixture[address] = MixtureModel(dists, weights)
        end
    end
    a2mixture
end
combine(v) = combine(v, ones(Float64, length(v)))

export reflate!, combine