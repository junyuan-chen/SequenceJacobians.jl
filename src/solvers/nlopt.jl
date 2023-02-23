# Wrap the objective function for tracing
function logposterior_nlopt_obj!(bm::BayesOrTrans, θ, grad, counter, printgap)
    l = logposterior_obj!(bm, θ, grad)
    counter[] += 1
    iter = counter[]
    if printgap > 0 && (iter-1) % printgap == 0
        @printf "iter %4i:  l = %10f" iter l
        println("  θ = ", θ)
    end
    return l
end

function _printgap(verbose::Union{Bool,Integer})
    if verbose === true
        return 20
    elseif verbose === false
        return 0
    else
        return Int(verbose)
    end
end

function SequenceJacobians.mode(bm::BayesOrTrans, solver::Symbol, θ0::AbstractVector;
        verbose::Union{Bool,Integer}=false, kwargs...)
    opt = NLopt.Opt(solver, dimension(bm))
    counter = Ref(0)
    printgap = _printgap(verbose)
    f(x, grad) = logposterior_nlopt_obj!(bm, x, grad, counter, printgap)
    opt.max_objective = f
    for (k, v) in kwargs
        setproperty!(opt, k, v)
    end
    # NLopt does not impose any stopping criterion by default
    haskey(kwargs, :ftol_abs) || (opt.ftol_abs = 1e-8)
    haskey(kwargs, :maxeval) || (opt.maxeval = 10_000)
    r = NLopt.optimize(opt, θ0)
    # Solver result may not be at the last evaluation left in bm
    p = parent(bm)
    _update_paravals!(p, bm isa BayesianModel ? r[2] : transform(bm.transformation, r[2]))
    return p[], r[2], counter[], r
end
