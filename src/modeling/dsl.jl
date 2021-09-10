const fn_primitives = [:sample, :observe, :transform, :plate]

get_fname_from_expr(f) = f.args[1].args[1]  # function name, the function has to be defined function ... end

function expand_trace_statements(__ex, primitives)
    if @capture(__ex, function_name_(xs__))
        function_name in primitives ? :($function_name(t, $(xs...))) : __ex
    else
        __ex
    end
end

function expand_statements(__ex, primitives)
    if @capture(__ex, varname_ = function_name_(xs__))
        if function_name in primitives
            if function_name != :plate
                :($varname = $function_name(t, $(string(varname)), $(xs...)))
            else
                # plate(t, op, address, ...) <- plate(op,...) 
                :($varname = $function_name(t, $(xs[1]), $(string(varname)), $(xs[2:end]...)))
            end
        else
            __ex
        end  
    else
        __ex
    end
end

@doc raw"""
    macro model(f)

Create a valid `CrimsonSkyline` model from a dsl that omits passthrough of the `Trace` data structure
and obviates address creation. For example, the model 
```
function f(t::Trace, data::Vector{Float64})
    loc = sample(t, "loc", Normal())
    log_scale = sample(t, "log_scale", Normal())
    plate(t, observe, "data", Normal(loc, exp(log_scale)), data)
end
```
can be created instead by writing
```
@model function simple_model(data::Vector{Float64})
    loc = sample(Normal())
    log_scale = sample(Normal())
    data = plate(observe, Normal(loc, exp(log_scale)), data)
end
```
The resulting function `f` is exactly equal to the first definition. 
The LHS of assignments *must* be parsed as `Symbols`, e.g., `my_variable = ...` is always safe;
`Expr`s like `out[i] = ...` won't _necessarily_ parse correctly.
The address of each site is equal to the string representation of the LHS of the assignment. 
For example, `log_scale = sample(Normal())` means that there will be a site called `"log_scale"`
in the trace.
"""
macro model(f)
    fname = get_fname_from_expr(f)
    f = prewalk(x -> expand_statements(x, fn_primitives), f)
    f = prewalk(x -> expand_trace_statements(x, [fname]), f)
    return quote
        $(esc(fname)) = $f
    end
end

export @model