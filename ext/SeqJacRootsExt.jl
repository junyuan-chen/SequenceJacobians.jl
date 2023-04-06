module SeqJacRootsExt

if isdefined(Base, :get_extension)
    using Roots
    using SequenceJacobians
else
    using ..Roots
    using ..SequenceJacobians
end

const SJ = SequenceJacobians

# Abstract types are renamed in Roots v2 with old names kept as alias
const AcceptedRootsMethod = Union{<:Roots.AbstractBracketing, <:Roots.AbstractSecant}

SJ.isscalarrootsolver(::AcceptedRootsMethod) = true
SJ.isscalarrootsolver(::Type{<:AcceptedRootsMethod}) = true
SJ.isscalarrootsolver(::Roots_Default) = true
SJ.isscalarrootsolver(::Type{Roots_Default}) = true

_solve(::Type{T}, f, x0; kwargs...) where T <: AcceptedRootsMethod =
    Roots.solve(Roots.ZeroProblem(f, x0), T(); kwargs...)

_solve(M::AcceptedRootsMethod, f, x0; kwargs...) =
    Roots.solve(Roots.ZeroProblem(f, x0), M; kwargs...)

_solve(::Type{Roots_Default}, f, x0; kwargs...) =
    Roots.solve(Roots.ZeroProblem(f, x0); kwargs...)

_solve(s::Roots_Default, f, x0; kwargs...) =
    solve(typeof(s), f, x0; kwargs...)

function SJ._solve!(ST::Type{T}, ss::SteadyState{TF}; x0=nothing, kwargs...) where
        {T<:Union{AcceptedRootsMethod, Roots_Default}, TF<:AbstractFloat}
    inlength(ss) == tarlength(ss) == 1 || throw(ArgumentError(
        "$ST is not accepted for multi-dimensional problems"))
    if T <: Roots.AbstractBracketing
        x0 === nothing && throw(ArgumentError(
            "must provide a bracket via positional argument x0"))
    else
        x0 === nothing && (x0 = ss.inits[1])
    end
    r = _solve(ST, ss, x0; kwargs...)::TF
    ss.inits[1] = r
    return ss[], r, !isnan(r)
end

end # module
