@doc raw"""
    function make_kernel(f, node::N, d::D) where {N<:Node, D<:ContinuousUnivariateDistribution}

Creates a single site proposal kernel for `node` that has an associated continuous univariate distribution.
The resulting proposal kernel is a normal distribution, ``z' | z \sim \mathrm{Normal}(z, (0.5 \sigma)^2)`` where
``\sigma`` is the standard deviation of the node's distribution.
"""
function make_kernel(f, node::N, d::D) where {N<:Node, D<:ContinuousUnivariateDistribution}
    ex = extrema(node.dist)
    sd = std(node.dist)
    function k(t0::Trace, t1::Trace, params...)
        n = truncated(Normal(t0[node.address].value, sd * 0.5), ex[1], ex[2])
        propose(t1, node.address, n)
    end
    k
end

@doc raw"""
    function make_kernel(f, node::N, d::D) where {N<:Node, D<:DiscreteUnivariateDistribution}

Creates a single site proposal kernel for `node` that has an associated discrete univariate distribution.
This method assumes that the distribution is defined over some subset of the integers; a support that does not meet 
this criteria *may* result in runtime inference errors. The resulting proposal kernel is a discrete nonparametric 
distribution, ``z' | z \sim \mathrm{DiscreteNonParametric}([z - 1, z, z + 1])`` if ``z`` is not on the boundary of 
the distribution's support; or ``z' | z \sim \mathrm{DiscreteNonParametric}([z \pm 1, z])`` if ``z`` is on the 
boundary of the distribution's support, with the sign depending on the left or right of the support.
"""
function make_kernel(f, node::N, d::D) where {N<:Node, D<:DiscreteUnivariateDistribution}
    ex = extrema(node.dist)
    make_kernel(f, node, d, ex, Val(ex[2]))
end

function make_kernel(f, node::N, d::D, ex, v) where {N<:Node, D<:DiscreteUnivariateDistribution}
    function k(t0::Trace, t1::Trace, params...)
        last = t0[node.address].value
        left = last - 1
        right = last + 1
        if last == ex[1]
            propose(t1, node.address, DiscreteNonParametric([last, right], [0.5, 0.5]))
        elseif last == ex[2]
            propose(t1, node.address, DiscreteNonParametric([left, last], [0.5, 0.5]))
        else
            w = ones(3)
            w = w ./ sum(w)
            propose(t1, node.address, DiscreteNonParametric([left, last, right], w))
        end
    end
    k
end

function make_kernel(f, node::N, d::D, ex, ::Val{Inf}) where {N<:Node, D<:DiscreteUnivariateDistribution}
    ex = extrema(node.dist)
    k(t0::Trace, t1::Trace, params...) = propose(t1, node.address, 
        truncated(Poisson(t0[node.address].value), ex[1], 2 * t0[node.address].value)
    )
    k
end

@doc raw"""
    function make_kernel(f, node::N, d::D) where {N<:Node, D<:ContinuousMultivariateDistribution}

Creates a single site proposal kernel for `node` that has an associated continuous multivariate distribution. 
Assumes that this site is *unconstrained* in ``\mathbb{R}^N``; using this method will result in inference runtime errors if the site
is constrained. The resulting proposal kernel is a multivariate normal,
``z' | z \sim \mathrm{MultivariateNormal}(z, 1 / \sqrt{|D|})`` where ``D`` is the dimensionality of the site. 
"""
function make_kernel(f, node::N, d::D) where {N<:Node, D<:ContinuousMultivariateDistribution}
    dim = size(node.dist)[1]
    k(t0::Trace, t1::Trace, params...) = propose(t1, node.address, MvNormal(t0[node.address].value, 1.0 / sqrt(dim)))
    k
end

@doc raw"""
    function make_kernel(f, node::N, d) where {N<:Node}

Create a single site proposal kernel for an arbitrary site. This method draws from the prior site distribution
and is therefore inefficient (both computationally and statistically). 
"""
function make_kernel(f, node::N, d) where {N<:Node}
    function k(t0::Trace, t1::Trace, params...)
        t = trace()
        f(t, params...)
        propose(t1, node.address, t[node.address].dist)
    end
    k
end

make_kernel(f, node::N) where N <: Node = make_kernel(f, node, node.dist)

@doc raw"""
    function make_kernels(f, t::Trace, addresses::Vector{T}; params = (), include::Bool=true) where T

Create a vector of single-site proposal functions for the stochastic function `f`. 
This method assumes static model structure and generates a proposal either for each site in `addresses` (`include = true`)
or for each site *not* in `addresses` (`include = false`). The tuple of params are any necessary parameters needed to execute
`f`.
"""
function make_kernels(f, t::Trace, addresses::Vector{T}; params = (), include::Bool=true) where T
    t = trace()
    f(t, params...)
    kernels = []
    cond = (a, addresses) -> include ? a in addresses : !(a in addresses)
    for (a, n) in t.trace
        if cond(a, addresses) && !t.trace[a].observed
            push!(kernels, make_kernel(f, n))
        end
    end
    kernels
end

@doc raw"""
    function make_kernels(f; params = ())

Create a vector of single-site proposal functions for the stochastic function `f`. 
This method assumes static model structure and generates a proposal kernel for each
model site. The tuple of params are any necessary parameters needed to execute
`f`.
"""
function make_kernels(f; params = ())
    t = trace()
    f(t, params...)
    a = collect(keys(t))
    make_kernels(f, t, a; params = params)
end

export make_kernel, make_kernels