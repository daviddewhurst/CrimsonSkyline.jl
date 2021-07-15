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

### q is a proposal kernel

function mh_step(rf::RandomField, q, x, prob_rf_x)
    x_prime_given_x = q(x)
    prob_x_prime_given_x = q(x_prime_given_x, x)
    prob_rf_x_prime = rf(x_prime_given_x)
    prob_x_given_x_prime = q(x, x_prime_given_x)
    log_a = prob_rf_x_prime + prob_x_given_x_prime - prob_rf_x - prob_x_prime_given_x
    accept(x, prob_rf_x, x_prime_given_x, prob_x_given_x_prime, log_a)
end
export mh_step

