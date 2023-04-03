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
"""
function rootsolvercache end

"""
    root(solver)

Retrieve the solution vector from `solver`.
"""
function root end

"""
    rootisfound(solver)

Determine whether the solver has found the solution successfully.
"""
function rootisfound end
