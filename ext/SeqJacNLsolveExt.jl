module SeqJacNLsolveExt

if isdefined(Base, :get_extension)
    using NLsolve
    using SequenceJacobians
    using SplitApplyCombine: splitdimsview
else
    using ..NLsolve
    using ..SequenceJacobians
end

const SJ = SequenceJacobians

SJ.isvectorrootsolver(::SJ.AbstractNLsolveSolver) = true
SJ.isvectorrootsolver(::Type{<:SJ.AbstractNLsolveSolver}) = true
SJ.isrootsolvercache(::NLsolve_Cache) = true

function SJ.rootsolvercache(::Type{NLsolve_newton}, ss::SteadyState; autodiff=:central, kwargs...)
    f!(y, x) = residuals!(y, ss, x)
    x0 = ss.inits
    df = NLsolve.OnceDifferentiable(f!, x0, copy(x0); autodiff=autodiff, inplace=true)
    ca = NLsolve.NewtonCache(df)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function SJ.rootsolvercache(::Type{NLsolve_trust_region}, ss::SteadyState;
        autodiff=:central, kwargs...)
    f!(y, x) = residuals!(y, ss, x)
    x0 = ss.inits
    df = NLsolve.OnceDifferentiable(f!, x0, copy(x0); autodiff=autodiff, inplace=true)
    ca = NLsolve.NewtonTrustRegionCache(df)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function SJ.rootsolvercache(::Type{NLsolve_anderson}, ss::SteadyState; m::Integer=1, kwargs...)
    # Only works for fixed point problems
    f!(y, x) = (residuals!(y, ss, x); y .-= x)
    x0 = ss.inits
    df = NLsolve.NonDifferentiable(f!, x0, copy(x0); inplace=true)
    ca = NLsolve.AndersonCache(df, m)
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function SJ.rootsolvercache(::Type{NLsolve_broyden}, ss::SteadyState; kwargs...)
    f!(y, x) = residuals!(y, ss, x)
    x0 = ss.inits
    df = NLsolve.OnceDifferentiable(f!, x0, copy(x0); inplace=true)
    ca = BroydenCache()
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

# Solver cache with default method and parameters
SJ.rootsolvercache(::Type{NLsolve_Solver}, ss::SteadyState; kwargs...) =
    rootsolvercache(NLsolve_trust_region, ss; kwargs...)

SJ.rootsolvercache(s::SJ.AbstractNLsolveSolver, ss::SteadyState; kwargs...) =
    rootsolvercache(typeof(s), ss; kwargs...)

function SJ.solve!(ca::NLsolve_Cache{CA}, x0;
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
function SJ.solve(::Type{NLsolve_Solver}, f!, x0; kwargs...)
    r = NLsolve.nlsolve(f!, x0; kwargs...)
    return r.zero, NLsolve.converged(r)
end

SJ.solve(s::NLsolve_Solver, f!, x0; kwargs...) = SJ.solve(typeof(s), f!, x0; kwargs...)

function backfunc!(y, x, ha, invals)
    backward_endo!(ha, splitdimsview(x)..., invals...)
    backward_exog!(ha)
    k = 1
    @inbounds for ev in SJ.expectedvalues(ha)
        @simd for i in eachindex(ev)
            # The order matters
            y[k] = ev[i] - x[k]
            k += 1
        end
    end
end

function SJ.backwardsolvercache(::Type{NLsolve_anderson}, b::HetBlock, invals::Tuple;
        m::Integer=1, kwargs...)
    backward_exog!(b.ha)
    evs = SJ.expectedvalues(b.ha)
    x0 = Array{eltype(getdist(b.ha)),ndims(evs[1])+1}(undef, size(evs[1])..., length(evs))
    f!(y, x) = backfunc!(y, x, b.ha, invals)
    df = NLsolve.NonDifferentiable(f!, x0, similar(x0); inplace=true)
    ca = NLsolve.AndersonCache(df, m)
    # Set initial values
    x0 = ca.x
    k = 1
    evs = SJ.expectedvalues(b.ha)
    @inbounds for ev in SJ.expectedvalues(b.ha)
        @simd for i in eachindex(ev)
            x0[k] = ev[i]
            k += 1
        end
    end
    return NLsolve_Cache{typeof(ca),typeof(df)}(ca, df)
end

function SJ._backward!(ca::NLsolve_Cache{<:NLsolve.AndersonCache}, b::HetBlock,
        invals::Tuple, maxbackiter, backtol, verbose, beta, aastart)
    # Update invals
    f!(y, x) = backfunc!(y, x, b.ha, invals)
    ca.df.f = f!
    backconverged = anderson!(ca, b, true, maxbackiter, backtol, verbose, beta, aastart)
    # Copy EVs back to ha
    k = 1
    @inbounds for ev in SJ.expectedvalues(b.ha)
        @simd for i in eachindex(ev)
            ev[i] = ca.ca.x[k]
            k += 1
        end
    end
    return backconverged
end

function SJ.backward!(::Type{NLsolve_anderson}, b::HetBlock, invals::Tuple)
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
    return SJ._backward!(ca, b, invals, maxbackiter, backtol, verbose, beta, aastart)
end

function forfunc!(y, x, ha)
    D = getdist(ha)
    Dendo = getdistendo(ha)
    forward!(Dendo, x, SJ.endoprocs(ha)...)
    forward!(D, Dendo, SJ.exogprocs(ha)...)
    @inbounds @simd for i in eachindex(y)
        y[i] = D[i] - x[i]
    end
end

function SJ.forwardsolvercache(::Type{NLsolve_anderson}, b::HetBlock; m::Integer=1, kwargs...)
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
    SJ.initendo!(b.ha)
    forconverged = anderson!(ca, b, false, maxforiter, fortol, verbose, beta, aastart)
    return forconverged
end

function SJ.forward!(::Type{NLsolve_anderson}, b::HetBlock, invals::Tuple)
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

# Modified from NLsolve.jl/src/solvers/anderson.jl
# for integration with value function iteration

@views function anderson!(ca::NLsolve_Cache, b::HetBlock,
        backward::Bool, maxiter::Integer, tol::Real, verbose::Bool, beta::Real=0.7,
        aa_start::Integer=100;
        store_trace::Bool=false,
        show_trace::Bool=false,
        extended_trace::Bool=false,
        droptol::Real=1e10, kwargs...)

    df = ca.df
    cache = ca.ca
    ha = b.ha
    converged = false

    tr = NLsolve.SolverTrace()
    tracing = store_trace || show_trace || extended_trace

    aa_iteration = cache.γs !== nothing
    m = aa_iteration ? length(cache.γs) : 0
    aa_iteration && (m_eff = 0)

    iter = 0
    while iter < maxiter
        iter += 1

        # evaluate function
        NLsolve.value!!(df, cache.x)
        fx = NLsolve.value(df)

        # check that all values are finite
        NLsolve.check_isfinite(fx)

        # compute next iterate of fixed-point iteration
        @. cache.g = cache.x + beta * fx

        # save trace
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(cache.x)
                dt["f(x)"] = copy(fx)
            end
            NLsolve.update!(tr, iter, maximum(abs, fx),
                iter > 1 ? NLsolve.sqeuclidean(cache.g, cache.x) :
                    convert(real(eltype(cache.x)), NaN),
                dt, store_trace, show_trace)
        end

        if backward
            st = backward_status(ha)
            st === nothing || verbose && println("    ", st)
            converged = backward_converged(ha, st, tol)
            converged && break
            foreach(x->copyto!(x[2], x[1]), backwardtargets(b.ha))
        else
            st = forward_status(ha)
            st === nothing || verbose && println("    ", st)
            converged = forward_converged(ha, st, tol)
            converged && break
            copyto!(getlastdist(ha), getdist(ha))
        end

        # update current iterate
        copyto!(cache.x, cache.g)

        # perform Anderson acceleration
        if aa_iteration
            if iter == aa_start
                # initialize cache of residuals and g
                copyto!(cache.fxold, fx)
                copyto!(cache.gold, cache.g)
            elseif iter > aa_start
                # increase size of history
                m_eff += 1

                # remove oldest history if maximum size is exceeded
                if m_eff > m
                    # circularly shift differences of g
                    ptr = cache.Δgs[1]
                for i in 1:(m-1)
                    cache.Δgs[i] = cache.Δgs[i + 1]
                end
                    cache.Δgs[m] = ptr

                    # delete left-most column of QR decomposition
                    NLsolve.qrdelete!(cache.Q, cache.R, m)

                    # update size of history
                    m_eff = m
                end

                # update history of differences of g
                @. cache.Δgs[m_eff] = cache.g - cache.gold

                # replace/add difference of residuals as right-most column to QR decomposition
                @. cache.fxold = fx - cache.fxold
                NLsolve.qradd!(cache.Q, cache.R, vec(cache.fxold), m_eff)

                # update cached values
                copyto!(cache.fxold, fx)
                copyto!(cache.gold, cache.g)

                # define current Q and R matrices
                Q, R = view(cache.Q, :, 1:m_eff), NLsolve.UpperTriangular(view(cache.R, 1:m_eff, 1:m_eff))

                # check condition (TODO: incremental estimation)
                if droptol > 0
                    while NLsolve.cond(R) > droptol && m_eff > 1
                        NLsolve.qrdelete!(cache.Q, cache.R, m_eff)
                        m_eff -= 1
                        Q, R = view(cache.Q, :, 1:m_eff), NLsolve.UpperTriangular(view(cache.R, 1:m_eff, 1:m_eff))
                    end
                end

                # solve least squares problem
                γs = view(cache.γs, 1:m_eff)
                NLsolve.ldiv!(R, NLsolve.mul!(γs, Q', vec(fx)))

                # update next iterate
                for i in 1:m_eff
                    @. cache.x -= cache.γs[i] * cache.Δgs[i]
                end
            end
        end
    end
    if store_trace
        b.ssargs[:trace] = tr
    end
    if !converged
        if backward
            @warn "backward iteration did not converge with tol=$tol after $iter steps"
        else
            @warn "forward iteration did not converge with tol=$tol after $iter steps"
        end
    end
    if converged && verbose
        if backward
            println("backward iteration converged with tol=$tol after $iter steps")
        else
            println("forward iteration converged with tol=$tol after $iter steps")
        end
    end
    return converged
end

end # module
