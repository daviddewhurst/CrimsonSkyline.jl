@doc raw"""
    function rejection(f, log_l :: Float64, params...)

Samples from the prior with the hard likelihood constraint ``\log L_k > `` `log_l`.

Args:
+ `f` stochastic function. Must have signature `f(t :: Trace, params...)`
+ `log_l`: current log likelihood threshold 
+ `params`: any additional arguments to pass to `f`
"""
function rejection(f, log_l :: Float64, params...)
    this_log_l = log_l
    local new_t
    while this_log_l <= log_l
        new_t = trace()
        f(new_t, params...)
        this_log_l = loglikelihood(new_t)
    end
    (new_t, this_log_l)
end

export rejection