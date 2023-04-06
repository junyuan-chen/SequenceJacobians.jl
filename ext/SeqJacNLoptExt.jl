module SeqJacNLoptExt

if isdefined(Base, :get_extension)
    using NLopt
    using Printf
    using SequenceJacobians
else
    using ..NLopt
    using ..Printf
    using ..SequenceJacobians
end

const SJ = SequenceJacobians
const NLopt_Supported = BayesOrTrans

# Wrap the objective function for tracing
function nlopt_obj!(m::NLopt_Supported, θ, grad, counter, printgap)
    f = m(θ, grad)
    counter[] += 1
    iter = counter[]
    if printgap > 0 && (iter-1) % printgap == 0
        @printf "iter %4i:  f(θ) = %10f" iter f
        println("  θ = ", θ)
    end
    return f
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

function _solve!(m::NLopt_Supported, solver::Symbol, θ0, maxf::Bool; verbose, kwargs...)
    opt = NLopt.Opt(solver, length(θ0))
    counter = Ref(0)
    printgap = _printgap(verbose)
    f(x, grad) = nlopt_obj!(m, x, grad, counter, printgap)
    maxf ? (opt.max_objective = f) : (opt.min_objective = f)
    for (k, v) in kwargs
        setproperty!(opt, k, v)
    end
    # NLopt does not impose any stopping criterion by default
    haskey(kwargs, :ftol_abs) || (opt.ftol_abs = 1e-8)
    haskey(kwargs, :maxeval) || (opt.maxeval = 10_000)
    r = NLopt.optimize(opt, θ0)
    r[3] in (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED) ||
        @warn "NLopt solver status is $(r[3])"
    return r, counter[]
end

function SJ.mode(bm::BayesOrTrans, solver::Symbol, θ0::AbstractVector;
        verbose::Union{Bool,Integer}=false, kwargs...)
    r, counter = _solve!(bm, solver, θ0, true; verbose, kwargs...)
    # Solver result may not be at the last evaluation left in bm
    p = parent(bm)
    SJ._update_paravals!(p.paravals, bm isa BayesianModel ? r[2] :
        transform(bm.transformation, r[2]), p.paraaxis)
    return p[], r[2], counter, r
end

end # module
