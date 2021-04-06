struct CPT{L<:AbstractDict, D<:AbstractArray}
    dims :: L
    axes :: Dict{Any, Int64}
    labels :: Array{Dict, 1}
    values :: D
end

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

renormalize!(c :: CPT) = c.values ./= sum(c.values)
ix_from_coord(c :: CPT, coord :: Tuple) = [c.labels[i][pt] for (i, pt) in enumerate(coord)]

function Base.setindex!(c :: CPT, value, coord :: Tuple)
    ix = ix_from_coord(c, coord)
    c.values[ix...] = value
end
Base.getindex(c :: CPT, coord :: Tuple) = c.values[ix_from_coord(c, coord)...]

export CPT, cpt, renormalize!