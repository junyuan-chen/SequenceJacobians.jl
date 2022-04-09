export NLsolve_newton, NLsolve_trust_region, NLsolve_anderson, NLsolve_broyden,
    NLsolve_Solver, NLsolve_Cache

abstract type AbstractNLsolveSolver end

isvectorrootsolver(::AbstractNLsolveSolver) = true
isvectorrootsolver(::Type{<:AbstractNLsolveSolver}) = true

struct NLsolve_newton <: AbstractNLsolveSolver end
struct NLsolve_trust_region <: AbstractNLsolveSolver end
struct NLsolve_anderson <: AbstractNLsolveSolver end
struct NLsolve_broyden <: AbstractNLsolveSolver end
struct NLsolve_Solver <: AbstractNLsolveSolver end

const Objective = Union{NLsolve.NonDifferentiable, NLsolve.OnceDifferentiable}

# Placeholder for Broyden method that does not require cache
struct BroydenCache <: NLsolve.AbstractSolverCache end

struct NLsolve_Cache{CA<:NLsolve.AbstractSolverCache, DF<:Objective}
    ca::CA
    df::DF
end

isrootsolvercache(::NLsolve_Cache) = true

function rootsolvercache(::Type{NLsolve_newton}, ss::SteadyState; autodiff=:central, kwargs...)
    f!(y, x) = residuals!(y, ss, x)
    x0 = ss.inits
    df = NLsolve.OnceDifferentiable(f!, x0, copy(x0); autodiff=autodiff, inplace=true)
    ca = NLsolve.NewtonCache(df)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function rootsolvercache(::Type{NLsolve_trust_region}, ss::SteadyState;
        autodiff=:central, kwargs...)
    f!(y, x) = residuals!(y, ss, x)
    x0 = ss.inits
    df = NLsolve.OnceDifferentiable(f!, x0, copy(x0); autodiff=autodiff, inplace=true)
    ca = NLsolve.NewtonTrustRegionCache(df)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function rootsolvercache(::Type{NLsolve_anderson}, ss::SteadyState; m::Integer=1, kwargs...)
    # Only works for fixed point problems
    f!(y, x) = (residuals!(y, ss, x); y .-= x)
    x0 = ss.inits
    df = NLsolve.NonDifferentiable(f!, x0, copy(x0); inplace=true)
    ca = NLsolve.AndersonCache(df, m)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function rootsolvercache(::Type{NLsolve_broyden}, ss::SteadyState; kwargs...)
    f!(y, x) = residuals!(y, ss, x)
    x0 = ss.inits
    df = NLsolve.OnceDifferentiable(f!, x0, copy(x0); inplace=true)
    ca = BroydenCache()
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

# Solver cache with default method and parameters
rootsolvercache(::Type{NLsolve_Solver}, ss::SteadyState; kwargs...) =
    rootsolvercache(NLsolve_trust_region, ss; kwargs...)

rootsolvercache(s::AbstractNLsolveSolver, ss::SteadyState; kwargs...) =
    rootsolvercache(typeof(s), ss; kwargs...)

function solve!(ca::NLsolve_Cache{CA}, x0;
        xtol::Real = zero(real(eltype(x0))),
        ftol::Real = convert(real(eltype(x0)), 1e-8),
        iterations::Integer = 1_000,
        store_trace::Bool = false,
        show_trace::Bool = false,
        extended_trace::Bool = false,
        linesearch = NLsolve.LineSearches.Static(),
        linsolve = (x, A, b) -> copyto!(x, A\b),
        factor::Real = one(real(eltype(x0))),
        autoscale::Bool = true,
        beta::Real = 1,
        aa_start::Integer = 1,
        droptol::Real = convert(real(eltype(x0)), 1e10)) where CA

    if show_trace
        NLsolve.@printf "Iter     f(x) inf-norm    Step 2-norm \n"
        NLsolve.@printf "------   --------------   --------------\n"
    end
    if CA <: NLsolve.NewtonCache
        r = NLsolve.newton(ca.df, x0, xtol, ftol, iterations,
            store_trace, show_trace, extended_trace, linesearch, ca.ca, linsolve=linsolve)
    elseif CA <: NLsolve.NewtonTrustRegionCache
        r = NLsolve.trust_region(ca.df, x0, xtol, ftol, iterations,
            store_trace, show_trace, extended_trace, factor, autoscale, ca.ca)
    elseif CA <: NLsolve.AndersonCache
        r = NLsolve.anderson(ca.df, x0, xtol, ftol, iterations,
            store_trace, show_trace, extended_trace, beta, aa_start, droptol, ca.ca)
    elseif CA <: BroydenCache
        r = NLsolve.broyden(ca.df, x0, xtol, ftol, iterations,
            store_trace, show_trace, extended_trace, linesearch)
    end
    return r.zero, NLsolve.converged(r)
end

# May specify solver with the method keyword
function solve!(::Type{NLsolve_Solver}, f!, x0; kwargs...)
    r = NLsolve.nlsolve(f!, x0; kwargs...)
    return r.zero, NLsolve.converged(r)
end

solve!(s::NLsolve_Solver, f!, x0; kwargs...) = solve!(typeof(s), f!, x0; kwargs...)
