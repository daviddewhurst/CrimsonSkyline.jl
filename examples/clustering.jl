import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Normal, truncated, Poisson, InverseWishart, MvNormal, Dirichlet, Categorical
using Logging
using StatsBase: mean, std
using Random: seed!
using PrettyPrint: pprintln

seed!(2021)

function eye(d :: Int)
    mat = zeros(Float64, d, d)
    for i in 1:d
        mat[i, i] = 1.0
    end
    mat
end

function clustering(t :: Trace, data :: Array{Float64, 2}, cluster_guess :: Int)
    # data has shape (D, N), where D is dimensionalty and N is num observations
    D = size(data, 1)
    N = size(data, 2)
    # first, choosing the number of clusters. Prior penalizes high number of clusters
    cluster_guess = max(cluster_guess, 1)
    n_clusters = sample(t, :n_clusters, truncated(Poisson(cluster_guess), 1, Inf))
    # second, given the number of clusters, define loc and cov for each
    covs = Array{Array{Float64, 2}, 1}()
    locs = Array{Array{Float64, 1}, 1}()
    id_mat = eye(D)
    for c in 1:n_clusters
        cov = sample(t, (:cov, c), InverseWishart(D + 1, id_mat))
        push!(covs, id_mat)
        loc = sample(t, (:loc, c), MvNormal(zeros(D), cov))
        push!(locs, loc)
    end
    # third, for each datapoint, assign to a cluster
    # note that each datapoint has its own local rv -- we could introduce an amortized model
    # that learns a mapping from datapoint features to cluster
    for n in 1:N
        cluster_prob = sample(t, (:cluster_prob, n), Dirichlet(n_clusters, 1.0))
        this_cluster = sample(t, (:this_cluster, n), Categorical(cluster_prob))
        observe(t, (:data, n), MvNormal(locs[this_cluster], covs[this_cluster]), data[:, n])
    end
end


function main()
    @info "Open-universe clustering inference"
    D = 4
    n1 = 5
    n2 = 12
    data1 = rand(D, n1)
    data2 = rand(D, n2) .+ 6
    data = hcat(data1, data2)
    @info "True number of clusters is 2"
    # pretend we don't know dgp, bias it to be higher than ground truth
    n_clusters_guess = 5
    @time results = mh(clustering; params = (data, n_clusters_guess), burn=2000, thin=100, num_iterations=100000)
    mean_n_clusters = mean(results, :n_clusters)
    std_n_clusters = std(results, :n_clusters)
    @info "posterior n_clusters ≈ $mean_n_clusters ± $std_n_clusters"
end

main()