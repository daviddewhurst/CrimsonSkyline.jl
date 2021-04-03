function logsumexp(w)
    offset = maximum(w)
    exp_w = exp.(w .- offset)
    s = sum(exp_w)
    log(s) + offset
end