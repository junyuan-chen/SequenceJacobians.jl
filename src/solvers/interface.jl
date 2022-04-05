"""
    AbstractRootSolver

Supertype for any root solver.
"""
abstract type AbstractRootSolver end

"""
    NoRootSolver <: AbstractRootSolver

A special type for indicating no root solver is needed.
"""
struct NoRootSolver <: AbstractRootSolver end

"""
    AbstractVectorRootSolver

Supertype for any multi-dimensional root solver for systems of equations.
"""
abstract type AbstractVectorRootSolver <: AbstractRootSolver end

"""
    AbstractScalarRootSolver

Supertype for any root solver that only solves a single equation with a single real variable.
"""
abstract type AbstractScalarRootSolver <: AbstractRootSolver end

"""
    AbstractSolverCache

Supertype for any solver cache.
"""
abstract type AbstractSolverCache end

"""
    solve!(SolverType::Type{<:AbstractRootSolver}, f, x0; kwargs...)
    solve!(cache::AbstractSolverCache, f, x0; kwargs...)

Solve the system of equations defined by function `f` with the specified solver type
with initial value `x0`.

If `SolverType` is a subtype of [`AbstractVectorRootSolver`](@ref),
the function `f` must takes two array arguments,
with the first being the reference to the residuals of the equations
and the second being the candidate of a root `x`;
`x0` is required to be an array.

If there is only one equation for a single unknown variable,
`SolverType` may be a subtype of [`AbstractScalarRootSolver`](@ref).
In this case, the function `f` only takes the root candidate `x` as the only argument
and `x0` is required to be a `Real`.
"""
function solve! end
