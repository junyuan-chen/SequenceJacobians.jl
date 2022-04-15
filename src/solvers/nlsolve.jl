export NLsolve_newton, NLsolve_trust_region, NLsolve_anderson, NLsolve_broyden,
    NLsolve_Solver, NLsolve_Cache, backwardsolvercache, forwardsolvercache

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
        droptol::Real = convert(real(eltype(x0)), 1e10),
        verbose::Bool=false, kwargs...) where CA

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
    converged = NLsolve.converged(r)
    converged ||
        @warn "solver did not converge with ftol=$(r.ftol) after $(r.iterations) steps"
    converged && verbose &&
        println("solver converged with ftol=$(r.ftol) after $(r.iterations) steps")
    return r.zero::typeof(x0), converged
end

# May specify solver with the method keyword
function solve!(::Type{NLsolve_Solver}, f!, x0; kwargs...)
    r = NLsolve.nlsolve(f!, x0; kwargs...)
    return r.zero, NLsolve.converged(r)
end

solve!(s::NLsolve_Solver, f!, x0; kwargs...) = solve!(typeof(s), f!, x0; kwargs...)

function backfunc!(y, x, ha, invals)
    backward_endo!(ha, splitdimsview(x)..., invals...)
    backward_exog!(ha)
    k = 1
    @inbounds for ev in expectedvalues(ha)
        @simd for i in eachindex(ev)
            # The order matters
            y[k] = ev[i] - x[k]
            k += 1
        end
    end
end

function backwardsolvercache(::Type{NLsolve_anderson}, b::HetBlock, invals::Tuple;
        m::Integer=1, kwargs...)
    backward_exog!(b.ha)
    evs = expectedvalues(b.ha)
    x0 = Array{eltype(getdist(b.ha)),ndims(evs[1])+1}(undef, size(evs[1])..., length(evs))
    f!(y, x) = backfunc!(y, x, b.ha, invals)
    df = NLsolve.NonDifferentiable(f!, x0, similar(x0); inplace=true)
    ca = NLsolve.AndersonCache(df, m)
    # Set initial values
    x0 = ca.x
    k = 1
    evs = expectedvalues(b.ha)
    @inbounds for ev in expectedvalues(b.ha)
        @simd for i in eachindex(ev)
            x0[k] = ev[i]
            k += 1
        end
    end
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function _backward!(ca::NLsolve_Cache{<:NLsolve.AndersonCache}, b::HetBlock,
        invals::Tuple, maxbackiter, backtol, verbose, beta, aastart)
    # Update invals
    f!(y, x) = backfunc!(y, x, b.ha, invals)
    ca.df.f = f!
    backconverged = anderson!(ca, b, true, maxbackiter, backtol, verbose, beta, aastart)
    # Copy EVs back to ha
    k = 1
    @inbounds for ev in expectedvalues(b.ha)
        @simd for i in eachindex(ev)
            ev[i] = ca.ca.x[k]
            k += 1
        end
    end
    return backconverged
end

function backward!(::Type{NLsolve_anderson}, b::HetBlock, invals::Tuple)
    if haskey(b.ssargs, :backwardsolvercache)
        ca = b.ssargs[:backwardsolvercache]::NLsolve_Cache{<:NLsolve.AndersonCache}
    else
        m = get(b.ssargs, :mbackward, 1)::Int
        ca = backwardsolvercache(NLsolve_anderson, b, invals, m=m)
        b.ssargs[:backwardsolvercache] = ca
    end
    verbose = haskey(b.ssargs, :verbose) ? b.ssargs[:verbose]>0 : false
    maxbackiter = get(b.ssargs, :maxbackiter, 5000)
    backtol = get(b.ssargs, :backtol, 1e-8)
    beta = get(b.ssargs, :backbeta, 0.7)
    aastart = get(b.ssargs, :backaastart, 100)
    return _backward!(ca, b, invals, maxbackiter, backtol, verbose, beta, aastart)
end

function forfunc!(y, x, ha)
    D = getdist(ha)
    Dendo = getdistendo(ha)
    forward!(Dendo, x, endoprocs(ha)...)
    forward!(D, Dendo, exogprocs(ha)...)
    @inbounds @simd for i in eachindex(y)
        y[i] = D[i] - x[i]
    end
end

function forwardsolvercache(::Type{NLsolve_anderson}, b::HetBlock; m::Integer=1, kwargs...)
    x0 = getdist(b.ha)
    f!(y, x) = forfunc!(y, x, b.ha)
    df = NLsolve.NonDifferentiable(f!, x0, copy(x0); inplace=true)
    ca = NLsolve.AndersonCache(df, m)
    # Set initial values
    copyto!(ca.x, x0)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function _forward!(ca::NLsolve_Cache{<:NLsolve.AndersonCache}, b::HetBlock,
        invals::Tuple, maxforiter, fortol, verbose, beta, aastart)
    initendo!(b.ha)
    forconverged = anderson!(ca, b, false, maxforiter, fortol, verbose, beta, aastart)
    return forconverged
end

function forward!(::Type{NLsolve_anderson}, b::HetBlock, invals::Tuple)
    if haskey(b.ssargs, :forwardsolvercache)
        ca = b.ssargs[:forwardsolvercache]::NLsolve_Cache{<:NLsolve.AndersonCache}
    else
        m = get(b.ssargs, :mforward, 1)::Int
        ca = forwardsolvercache(NLsolve_anderson, b, m=m)
        b.ssargs[:forwardsolvercache] = ca
    end
    verbose = haskey(b.ssargs, :verbose) ? b.ssargs[:verbose]>0 : false
    maxforiter = get(b.ssargs, :maxforiter, 5000)
    fortol = get(b.ssargs, :fortol, 1e-10)
    beta = get(b.ssargs, :forbeta, 0.7)
    aastart = get(b.ssargs, :foraastart, 100)
    return _forward!(ca, b, invals, maxforiter, fortol, verbose, beta, aastart)
end
