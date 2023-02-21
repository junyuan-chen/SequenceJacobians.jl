export Roots_Default

# Abstract types are renamed in Roots v2 with old names kept as alias
const AcceptedRootsMethod = Union{<:Roots.AbstractBracketing, <:Roots.AbstractSecant}

isscalarrootsolver(::AcceptedRootsMethod) = true
isscalarrootsolver(::Type{<:AcceptedRootsMethod}) = true

struct Roots_Default end

isscalarrootsolver(::Roots_Default) = true
isscalarrootsolver(::Type{Roots_Default}) = true

function solve!(::Type{T}, f, x0; kwargs...) where T <: AcceptedRootsMethod
    z = Roots.solve(Roots.ZeroProblem(f, x0), T(); kwargs...)
    return z, true
end

function solve!(M::AcceptedRootsMethod, f, x0; kwargs...)
    z = Roots.solve(Roots.ZeroProblem(f, x0), M; kwargs...)
    return z, true
end

function solve!(::Type{Roots_Default}, f, x0; kwargs...)
    z = Roots.solve(Roots.ZeroProblem(f, x0); kwargs...)
    return z, !isnan(z)
end

solve!(s::Roots_Default, f, x0; kwargs...) = solve!(typeof(s), f, x0; kwargs...)

function solve!(ST::Type{T}, ss::SteadyState{TF}; x0=nothing, kwargs...) where
        {T<:Union{AcceptedRootsMethod, Roots_Default}, TF<:AbstractFloat}
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
    return ss[]
end
