function replay(f :: F, t :: Trace) where F <: Function
    t_new = deepcopy(t)
    interpret_latent!(t_new, REPLAYED)
    g(params...) = f(t_new, params...)
    (t_new, g)
end

function block(f :: F, t :: Trace, addresses) where F <: Function
    t_new = deepcopy(t)
    for a in addresses
        t_new[a].interpretation = BLOCKED
    end
    g(params...) = f(t_new, params...)
    (t_new, g)
end

block(f :: F, t :: Trace) where F <: Function = block(f, t, keys(t))

export block, replay