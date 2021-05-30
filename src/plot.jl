function plot_marginal(r::NonparametricSamplingResults{I}, a) where I<:InferenceType
    l = @layout([a{0.01h}; [b c]])
    title = plot(
        title = "Marginal $a posterior, computed using $(string(typeof(r.interpretation)))", 
        grid = false, 
        showaxis = false, 
        bottom_margin = -5Plots.px
    )
    p1 = histogram(
        r[a], normed=true, color=:black,
        xlabel=a, ylabel="p($a | data)", label=""
    )
    p2 = plot(
        r[a], color=:black,
        xlabel="Trace index", ylabel=a, label=""
    )
    plot(title, p1, p2, layout=l)
end

function plot_marginal(r::SamplingResults{I}, a, savepath::S, savename::S) where{I<:InferenceType, S<:AbstractString}
    mkpath(savepath)
    fname = joinpath(savepath, savename)
    p = plot_marginal(r, a)
    plot!(p, size=(600,400))
    savefig(p, fname)
end

export plot_marginal