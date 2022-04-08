export Roots_Solver, Roots_Default_Solver

struct Roots_Solver{T<:Union{Roots.AbstractUnivariateZeroMethod,Nothing}} <: AbstractScalarRootSolver end

const Roots_Default_Solver = Roots_Solver{Nothing}

function solve!(t::Type{Roots_Solver{T}}, f, x0; kwargs...) where T
    z = Roots.solve(Roots.ZeroProblem(f, x0), T(); kwargs...)
    return z, true
end

function solve!(t::Type{Roots_Default_Solver}, f, x0; kwargs...)
    z = Roots.solve(Roots.ZeroProblem(f, x0); kwargs...)
    return z, !isnan(z)
end

function solve!(ST::Type{Roots_Solver{T}}, ss::SteadyState{TF};
        x0=nothing, kwargs...) where {T,TF}
    inlength(ss) == tarlength(ss) == 1 || throw(ArgumentError(
        "$ST is not accepted for multi-dimensional problems"))
    function f(x)
        ss.inits[1] = x
        return residuals!(ss)[1]
    end
    if T <: Roots.AbstractBracketing
        x0 === nothing && throw(ArgumentError(
            "must provide a bracket via keyword argument x0"))
    else
        x0 === nothing && (x0 = ss.inits[1])
    end
    solve!(ST, f, x0; kwargs...)::Tuple{TF,Bool}
    return getvarvals(ss)
end
