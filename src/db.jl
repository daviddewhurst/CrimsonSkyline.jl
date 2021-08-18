function is_at_widest_option_type(x)
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

function get_types_from_iterable(x)
    types = Set()
    for v in x
        union!(types, Set((typeof(v),)))
    end
    collect(types)
end

function narrow_column_types!(data)
    for k in keys(data)
        types = get_types_from_iterable(data[k])
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
                push!(data[column_name], missing)
                push!(data[cnlp], missing)
            end
        end
    end
    data
end

function to_df(f; params = (), num_iterations = 100, method::InferenceType = FORWARD, inference_params = Dict())
    check_sf(f)
    inference_params["num_iterations"] = num_iterations
    results = inference(f, method; params = params, inference_params = inference_params)
    column_names = addresses(results)
    types_cnames = get_types_from_iterable(column_names)
    !((length(types_cnames) == 1) && (types_cnames[1] == String)) && error("$f does not have all string addresses")
    data = get_data(results, column_names)
    !all(map(v -> is_at_widest_option_type(v), values(data))) && error("Each column type is not typed at widest Option[T]")
    narrow_column_types!(data)
    to_df(data)
end
export to_df

abstract type TabularModel{F} end

struct DataFrameModel{F} <: TabularModel{F}
    f::F
    table::DataFrame
end
function DataFrameModel(f::F; params = (), num_iterations = 100, method::InferenceType = FORWARD, inference_params = Dict()) where F
    df = to_df(f; params = params, num_iterations = num_iterations, method = method, inference_params = inference_params)
    DataFrameModel(f, df)
end

struct SQLModel{F} <: TabularModel{F}
    f::F
    db::SQLite.DB
end
function SQLModel(f::F; params = (), num_iterations = 100, method::InferenceType = FORWARD, inference_params = Dict()) where F
    df = to_df(f; params = params, num_iterations = num_iterations, method = method, inference_params = inference_params)
    db = SQLite.DB()
    tablename = df |> SQLite.load!(db, "$(f)_results")
    @info "Created table $tablename"
    SQLModel(f, db)
end
export DataFrameModel, SQLModel

function query(sqm::SQLModel, q::String)
    DBInterface.execute(sqm.db, q) |> DataFrame
end
export query