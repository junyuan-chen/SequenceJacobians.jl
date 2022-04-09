"""
    NoRootSolver

A singleton type for indicating no root solver is needed.
"""
struct NoRootSolver end

"""
    isvectorrootsolver(s)

Determine whether `s` is a multi-dimensional root solver with a recognized interface
for solving systems of equations.
See also [`isscalarrootsolver`](@ref) and [`isrootsolver`](@ref).
"""
isvectorrootsolver(::Any) = false

"""
    isscalarrootsolver(s)

Determine whether `s` is a one-dimensional root solver with a recognized interface
for solving a single equation with a single real variable.
See also [`isvectorrootsolver`](@ref) and [`isrootsolver`](@ref).
"""
isscalarrootsolver(::Any) = false

"""
    isrootsolver(s)

Determine whether `s` is a root solver with a recognized interface.
See also [`isvectorrootsolver`](@ref), [`isscalarrootsolver`](@ref)
and [`isrootsolvercache`](@ref).
"""
isrootsolver(s::Any) = isvectorrootsolver(s) || isscalarrootsolver(s)

"""
    isrootsolvercache(ca)

Determine whether `ca` is a recognized cache for root solvers.
See also [`isrootsolver`](@ref).
"""
isrootsolvercache(ca::Any) = false

"""
    rootsolvercache(solver, ss::SteadyState; kwargs...)

Try to construct the cache object of `solver` for solving the steady state defined by `ss`.
The fallback method returns `nothing`.
"""
function rootsolvercache end

"""
    solve!(solver, f, x0; kwargs...)
    solve!(SolverType, f, x0; kwargs...)
    solve!(cache, x0; kwargs...)

Solve the system of equations defined by function `f`
with the specified `solver` or `SolverType`
with initial value `x0`.
If a cache object associated with the solver is available,
one may directly provide the `cache` instead.
With recognized objects,
[`isrootsolver`](@ref) returns `true` for `solver` or `SolverType`;
[`isrootsolvercache`](@ref) returns `true` for `cache`.

If [`isvectorrootsolver`](@ref) returns `true`,
the function `f` must takes two array arguments,
with the first being the reference to the residuals of the equations
and the second being the candidate of a root `x`;
`x0` is required to be an array.

If [`isscalarrootsolver`](@ref) returns `true`,
the function `f` only takes the root candidate `x` as the only argument
and `x0` is required to be a `Real`.
"""
function solve! end
