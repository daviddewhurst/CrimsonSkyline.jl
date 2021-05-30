@doc raw"""
    function aic(t :: Trace)

Computes the Akaike Information Criterion for a *single trace* (thus replacing the definition) with 
"maximum likelihood" by one with "likelihood". The formula is 

```math
\text{AIC}(t)/2 = |\text{params}(t)| - \ell(t),
```

where ``\text{params}(t)|`` is the sum of the dimensionalities of non-observed 
and non-deterministic sample nodes.
"""
function aic(t :: Trace)
    ll = 0.0
    k = 0
    for v in values(t)
        if !(v.interpretation == DETERMINISTIC || v.observed)
            length(size(v.value)) == 0 ? k += 1 : k += length(v.value)
        end
        if v.observed 
            ll += v.logprob_sum
        end
    end
    2.0 * (k - ll)
end

function numparams(t::Trace)
    k = 0
    for v in values(t)
        if !(v.interpretation == DETERMINISTIC || v.observed)
            length(size(v.value)) == 0 ? k += 1 : k += length(v.value)
        end
    end
    k
end

@doc raw"""
    function aic(r :: SamplingResults{I}) where I <: InferenceType

Computes an empirical estimate of the Akaike Information Criterion from a `SamplingResults`.
The formula is 
    
```math
\text{AIC}(r)/2 = \min_{t \in \text{traces}(r)}|\text{params}(t)| - \hat\ell(t),
```
    
where ``\text{params}(t)|`` is the sum of the dimensionalities of non-observed 
and non-deterministic sample nodes and
``\hat\ell(t)`` is the empirical maximum likelihood.
"""
function aic(r :: SamplingResults{I}) where I <: InferenceType
    min_aic = Inf
    for t in r.traces
        a = aic(t)
        if a < min_aic
            min_aic = a
        end
    end
    min_aic
end

@doc raw"""
    function hpds(r::SamplingResults{I}, pct::Float64) where I <: InferenceType

Computes the highest posterior density set (HPDS) of the `SamplingResults` object.
Let ``\mathcal T`` be the set of traces. The ``100\times Q \%``-percentile HPDS is defined as the 
set that satisfies ``\sum_{t \in \mathrm{HPDS}} p(t) = Q`` and, for all ``t \in \mathrm{HPDS}``,
``p(t) > p(s)`` for every ``s \in \mathcal T  - \mathrm{HPDS}``.
It is possible to compute the HPDS using the full joint density ``p(t) \equiv p(x, z)``, where ``x`` is the 
set of observed rvs and ``z`` is the set of latent rvs, since ``p(z|x) \propto p(x, z)``. 

`pct` should be a float in (0.0, 1.0). E.g., `pct = 0.95` returns the 95% HPDS.
"""
function hpds(r::SamplingResults{I}, pct::Float64) where I <: InferenceType
    (pct > 0.0 && pct < 1.0) || error("pct must be in (0, 1)")
    ix = Int(floor(length(r.traces) * pct))
    for t in r.traces
        if t.logprob_sum == 0.0
            logprob!(t)
        end
    end
    # p(z|x) \propto p(x, z), so we can sort by the joint density, which is easy to 
    # calculate, instead of needing to approximate the posterior density
    tr_ix = partialsortperm(r.traces, 1:ix, by=(t,) -> t.logprob_sum, rev=true)
    NonparametricSamplingResults{typeof(r.interpretation)}(
        r.interpretation,
        r.log_weights[tr_ix],
        r.return_values[tr_ix],
        r.traces[tr_ix]
    )
end

@doc raw"""
    function hpdi(r::SamplingResults{I}, pct::Float64, addresses::AbstractArray{T}) where {I <: InferenceType, T}

Computes the highest posterior density interval(s) for a univariate variable. Does *not* check that the 
data corresponding to each address in `addresses` is actually univariate; if in doubt, use `hpds` instead.
"""
function hpdi(r::SamplingResults{I}, pct::Float64, addresses::AbstractArray{T}) where {I <: InferenceType, T}
    r = hpds(r, pct)
    Dict(a => (minimum(r[a]), maximum(r[a])) for a in addresses)
end

export aic, hpds, hpdi, numparams