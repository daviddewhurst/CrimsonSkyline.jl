factor_ab(x) = logpdf(MvNormal(PDMat([1.0 0.5; 0.5 1.0])), x)
factor_bc(x) = logpdf(MvNormal([2.0, 2.0], PDMat([2.0 -1.0; -1.0 2.0])), x)

struct FactorProposal
    addresses :: Vector{String}
    last :: Vector{String}
    std :: Float64
end
FactorProposal(addresses) = FactorProposal(addresses, [addresses[1]], 0.5)
function (fp::FactorProposal)(x::Dict)
    new = deepcopy(x)
    address = rand(fp.addresses)
    new[address] = randn() * fp.std + new[address]
    fp.last[1] = address
    new
end
function (fp::FactorProposal)(x_prime::Dict, x::Dict)
    address = fp.last[1]
    logpdf(Normal(x[address], fp.std), x_prime[address]) - log(length(fp.addresses))
end

@testset "make field and sample" begin
    factors = Dict(["a", "b"] => factor_ab, ["b", "c"] => factor_bc)
    evidence = Dict("b" => 3.0)
    #evidence = Dict()
    field = RandomField(factors, evidence)
    addresses = ["a", "c"]
    proposal = FactorProposal(addresses)
    initial_values = Dict("a" => 0.0, "b" => 3.0, "c" => 0.0)
    samples = mh(field, [proposal], initial_values; burn=2500, thin=500, num_iterations=52500)
    for address in ["a", "b", "c"]
        plot_marginal(samples, address, "out", "factor-marginal-$address.png")
    end
end