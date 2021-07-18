# Random fields

@doc raw"""
    struct RandomField 
        names::Set{String}
        factors::Dict{Vector{String},Function}
        evidence::Dict{String,Any}
    end

A representation of a random field by a collection of factors: 
``\log p(x) = \sum_{f \in \mathcal F} \log \psi_f(x_f)``, where 
``\mathcal F`` is the set of (log) factors and ``x_f`` is the set of variables
incident on that (log) factor. `factors` should be properly normalized *log* mass or density
functions. There is no restriction on the state space of the variables involved as long
as the factor functions can evaluate the log probability of the variables.
For example, 
```
factor_ab(x) = logpdf(MvNormal(PDMat([1.0 0.5; 0.5 1.0])), x)
factor_bc(x) = logpdf(MvNormal([2.0, 2.0], PDMat([2.0 -1.0; -1.0 2.0])), x)
```
are two valid (log) factor functions -- the first corresponding to the factor ``\log \psi_{a,b}(x_a,x_b)``
and the second corresponding to the factor ``\log \psi_{b,c}(x_b, x_c)``.
Posting evidence is done using a dictionary mapping an address to value, 
e.g., `evidence = Dict("b" => 3.0)`.

Calling a random field corresponds to evaluating its log probability with the passed argument,
e.g., 
```
my_rf = RandomField(...)
x = Dict("a" => 1.0, "b" => -2.1)
my_lp = my_rf(x)  # corresponds to logprob(my_rf, x)
```
"""
struct RandomField 
    names::Set{String}
    factors::Dict{Vector{String},Function}
    evidence::Dict{String,Any}
end

struct GenerativeField{T}
    field::RandomField
    proposal::T
    val::Dict
    function GenerativeField{T}(field::RandomField, proposal::T, val::Dict) where T
        for (k,v) in field.evidence
            val[k] = v
        end
        new(field, proposal, val)
    end
end
GenerativeField(field::RandomField, proposal::T, val::Dict) where T = GenerativeField{T}(field::RandomField, proposal::T, val::Dict)
export GenerativeField

function Distributions.rand(gf::GenerativeField; num_iterations=2500)
    results = mh(
        gf.field, [gf.proposal], gf.val;
        burn = num_iterations, thin = 1, num_iterations = num_iterations + 1
    )
    sample = Dict(k => v[1] for (k,v) in results.values)
    for (k, v) in sample
        gf.val[k] = v
    end
    sample
end

@doc raw"""
    function logprob(rf::RandomField, x::Dict)

Evaluates the log probability of a set of values against the density described by the
random field `rf`. The values `x` should have the format `address => value`.
"""
function logprob(rf::RandomField, x::Dict)
    lp = 0.0
    evidence_vars = keys(rf.evidence)
    for (variables, factor) in rf.factors
        to_score = []
        for v in variables
            # safeguard proposing to observed components
            if v in evidence_vars
                push!(to_score, rf.evidence[v])
            else
                push!(to_score, x[v])
            end
        end
        lp += rf.factors[variables](to_score)
    end
    lp
end
(rf::RandomField)(x::Dict) = logprob(rf, x)
logprob(gf::GenerativeField, x::Dict) = logprob(gf.field, x)

@doc raw"""
    function RandomField(factors::Dict{Vector{String},Function})

Outer constuctor for `RandomField` that requires only a dict of factors.
"""
function RandomField(factors::Dict{Vector{String},Function})
    evidence = Dict{String,Any}()
    RandomField(factors, evidence)
end

@doc raw"""
    function RandomField(factors::Dict{Vector{String},Function}, evidence::Dict)

Outer constructor for `RandomField` that requires a dict of factors and allows posting
evidence when the field is created (instead of manually doing so later).
"""
function RandomField(factors::Dict{Vector{String},Function}, evidence::Dict)
    names = Set{String}()
    for k in keys(factors)
        union!(names, k)
    end
    RandomField(names, factors, evidence)
end

export RandomField, logprob

accept(x, prob_x, x_prime, prob_x_prime, log_a::Float64) = log(rand()) < log_a ? (x_prime, prob_x_prime) : (x, prob_x)
export accept