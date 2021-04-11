@doc raw"""
    struct CPT{L<:AbstractDict, D<:AbstractArray}
        dims :: L
        axes :: Dict{Any, Int64}
        labels :: Array{Dict, 1}
        values :: D
    end

A representation of an arbitrary-dimensional ragged CPT. Supports `Base` methods 
`setindex!` and `getindex`.
"""
struct CPT{L<:AbstractDict, D<:AbstractArray}
    dims :: L
    axes :: Dict{Any, Int64}
    labels :: Array{Dict, 1}
    values :: D
end

@doc raw"""
    function cpt(dims :: L) where L <: AbstractDict

Outer constructor for `CPT`. Given a `Dict` that maps names of dimensions to dimension level
names, constructs a CPT with equiprobable coordinates.

Example usage:
```
dims = Dict("cost" => ["high", "low"], "revenue" => ["high", "medium", "low"])
c = cpt(dims)
c[("high", "low")] = 0.4
c[("high", "high")] = 0.2
c[("low", "low")] = 0.3
renormalize!(c)
```
"""
function cpt(dims :: L) where L <: AbstractDict
    labels = Array{Dict, 1}()
    sizes = Array{Int64, 1}()
    axes = Dict{Any, Int64}()
    for (i, (dim, vs)) in enumerate(dims)
        push!(labels, Dict(v => i for (i, v) in enumerate(vs)))
        push!(sizes, length(labels[i]))
        axes[dim] = i
    end
    values = ones(Float64, sizes...) ./ prod(sizes)
    CPT{typeof(dims), typeof(values)}(dims, axes, labels, values)
end

@doc raw"""
    renormalize!(c :: CPT)

Renormalizes the possibly non-normalized factor `c` to be a proper discrete joint density.
"""
renormalize!(c :: CPT) = c.values ./= sum(c.values)

ix_from_coord(c :: CPT, coord :: Tuple) = [c.labels[i][pt] for (i, pt) in enumerate(coord)]

function Base.setindex!(c :: CPT, value, coord :: Tuple)
    ix = ix_from_coord(c, coord)
    c.values[ix...] = value
end
Base.getindex(c :: CPT, coord :: Tuple) = c.values[ix_from_coord(c, coord)...]

export CPT, cpt, renormalize!