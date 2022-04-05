export Roots_Solver, Roots_Default_Solver

struct Roots_Solver{T<:Union{Roots.AbstractUnivariateZeroMethod,Nothing}} <: AbstractScalarRootSolver end

const Roots_Default_Solver = Roots_Solver{Nothing}

function solve!(t::Type{Roots_Solver{T}}, f, x0; kwargs...) where T
    z = Roots.find_zero(f, x0, T(); kwargs...)
    return z, true
end

function solve!(t::Type{Roots_Default_Solver}, f, x0; kwargs...)
    z = Roots.find_zero(f, x0; kwargs...)
    return z, true
end

function solve!(ST::Type{Roots_Solver{T}}, ss::SteadyState; x0=nothing, kwargs...) where T
    length(ss.resids)==1 && length(ss.ins)==1 || throw(ArgumentError(
        "$ST is not accepted for multi-dimensional problems"))
    function f(x)
        ss.varvals[ss.ins[1]] = x
        return residuals!(ss)[1]
    end
    if T <: Roots.AbstractBracketing
        x0 === nothing && throw(ArgumentError(
            "must provide a bracket via keyword argument x0"))
    else
        x0 === nothing && (x0 = ss.inits[1])
    end
    return solve!(ST, f, x0; kwargs...)
end
