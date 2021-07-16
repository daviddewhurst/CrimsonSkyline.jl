# Random fields

struct RandomField 
    names::Set{String}
    factors::Dict{Vector{String},Function}
    evidence::Dict{String,Any}
end

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

function RandomField(factors::Dict{Vector{String},Function})
    evidence = Dict{String,Any}()
    RandomField(factors, evidence)
end

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