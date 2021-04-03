function interpret_latent!(t :: Trace, i :: Interpretation)
    for a in keys(t)
        if !t[a].observed
            t[a].interpretation = i
        end
    end
end

function replay(f :: F, t :: Trace, params...) where F <: Function
    t_new = deepcopy(t)
    interpret_latent!(t_new, Replayed())
    r = f(t_new, params...)
    interpret_latent!(t_new, Nonstandard())
    (r, t_new)
end

export replay