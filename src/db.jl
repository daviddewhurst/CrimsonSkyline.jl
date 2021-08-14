signatures(f) = map(m -> m.sig, methods(f).ms)
is_sf(f) = any(map(m -> Trace in m.parameters, signatures(f)))
export is_sf

function is_at_most_option_type(x)
    types = Set()
    for v in x
        union!(types, Set((typeof(v),)))
    end
    lt = length(types)
    lt > 2 && return false
    lt > 1 && return ((Nothing in types) || (Missing in types) ? true : false)
    true
end
export is_at_most_option_type

function get_types_from_any(x)
    types = Set()
    for v in x
        union!(types, Set((typeof(v),)))
    end
    collect(types)
end

function narrow_column_types!(data)
    for k in keys(data)
        types = get_types_from_any(data[k])
        TYPE = length(types) > 1 ? Union{types...} : types[1]
        data[k] = convert(Vector{TYPE}, data[k])
    end
end

to_df(data::Dict{String,Any}) = DataFrame(Any[values(data)...], [Symbol(k) for k in keys(data)])

function get_data(results, column_names)
    data = Dict{String, Any}()
    for column_name in column_names
        data[column_name] = []
        cnlp = column_name * "_logprob"
        data[cnlp] = []
        for (ix, trace) in enumerate(results.traces)
            kt = keys(trace)
            if column_name in kt
                push!(data[column_name], trace[column_name].value)
                push!(data[cnlp], trace[column_name].logprob_sum)
            else
                push!(data[column_name], nothing)
                push!(data[cnlp], nothing)
            end
        end
    end
    data
end

function table(f; params = (), num_iterations = 100)
    !is_sf(f) && error("$f does not appear to be a stochastic function.")
    results = forward_sampling(f; params = params, num_iterations = num_iterations)
    column_names = addresses(results)
    types_cnames = get_types_from_any(column_names)
    !((length(types_cnames) == 1) && (types_cnames[1] == String)) && error("$f does not have all string addresses")
    data = get_data(results, column_names)
    !all(map(v -> is_at_most_option_type(v), values(data))) && error("Each column type is not typed at widest Option[T]")
    narrow_column_types!(data)
    df = to_df(data)
    df
end
export table