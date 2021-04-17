to_julia(x) = eval(Meta.parse(x))

function to_column_vectors(t :: Trace)
    addresses = []
    dists = []
    _values = []
    logprobs = []
    logprobsums = []
    observeds = []
    interpretations = []
    last_interpretations = []

    for node in values(t)
        push!(addresses, node.address)
        push!(dists, node.dist)
        push!(_values, node.value)
        push!(logprobs, node.logprob)
        push!(logprobsums, node.logprob_sum)
        push!(observeds, node.observed)
        push!(interpretations, node.interpretation)
        push!(last_interpretations, node.last_interpretation)
    end
    [
        "address" => addresses, 
        "dist" => dists,
        "value" => _values,
        "logprob" => logprobs,
        "logprob_sum" => logprobsums,
        "observed" => observeds,
        "interpretation" => interpretations,
        "last_interpretation" => last_interpretations
    ]
end

function dataframe(t :: Trace)
    col_vecs = to_column_vectors(t)
    DataFrame(col_vecs...)
end

@doc raw"""
    function save(t :: Trace, f)

Saves a trace `t` at the path `f`, and returns the path `f`.
This works with `load`, so that `load(save(t, f)) == t`.
"""
function save(t :: Trace, f)
    df = dataframe(t)
    CSV.write(f, df)
end

export save
