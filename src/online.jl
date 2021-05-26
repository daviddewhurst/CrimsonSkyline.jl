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

function dist_dict(r::ParametricSamplingResults)
    dists = r.distributions
    d = Dict()
    for (a, dist) in dists
        d[a] = dist_dict(dist)
    end
    d
end
dist_dict(n::Normal) = Dict("dist" => "Normal","loc" => n.μ,"scale" => n.σ)
dist_dict(n::LogNormal) = Dict("dist" => "LogNormal","loc" => n.μ,"scale" => n.σ)
dist_dict(p::Poisson) = Dict("dist" => "Poisson","lambda" => p.λ)
dist_dict(d::DiscreteUniform) = Dict("dist" => "DiscreteUniform","a" => d.a,"b" => d.b)
dist_dict(m::MvNormal) = Dict("dist"=>"MvNormal", "loc"=>m.μ, "cov"=>m.Σ)
dist_dict(m::MvLogNormal) = Dict("dist"=>"MvLogNormal", "loc"=>m.normal.μ, "cov"=>m.normal.Σ)

to_json(r::ParametricSamplingResults) = JSON.json(Dict("distributions"=>dist_dict(r), "interpretation"=>string(r.interpretation)))

function dict_dist(dists::Dict)
    d = Dict()
    for (a, dist) in dists
        d[a] = dict_dist(dist, dist["dist"])
    end
    d
end
function dict_dist(d::Dict, s::String)
    if s == "Normal"
        Normal(d["loc"], d["scale"])
    elseif s == "LogNormal"
        LogNormal(d["loc"], d["scale"])
    elseif s == "Poisson"
        Poisson(d["lambda"])
    elseif s == "DiscreteUniform"
        DiscreteUniform(d["a"], d["b"])
    elseif s == "MvNormal"
        MvNormal(d["loc"], d["cov"])
    elseif s == "MvLogNormal"
        MvLogNormal(MvNormal(d["loc"], d["cov"]))
    end
end

function from_json(s::String)
    dict = JSON.parse(s)
    dists = dict_dist(dict["distributions"])
    I = eval(Meta.parse(dict["interpretation"]))
    ParametricSamplingResults{typeof(I)}(I, Float64[], [], Trace[], dists)
end

export reflate!, combine, to_json, from_json