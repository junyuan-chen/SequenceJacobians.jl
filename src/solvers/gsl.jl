export GSL_MultirootFSolver, GSL_MultirootFSolverCache, GSL_Hybrids

abstract type GSL_MultirootFSolver <: AbstractVectorRootSolver end

struct GSL_MultirootFSolverCache{V,S} <: AbstractSolverCache
    N::Int
    vinit::V
    solver::S
end

struct GSL_Hybrids <: GSL_MultirootFSolver end

function GSL_MultirootFSolverCache(::Type{GSL_Hybrids}, N::Int)
    vinit = GSL.vector_alloc(N)
    solver = GSL.multiroot_fsolver_alloc(GSL.gsl_multiroot_fsolver_hybrids, N)
    return GSL_MultirootFSolverCache{typeof(vinit),typeof(solver)}(N, vinit, solver)
end

function _setvinit!(ca::GSL_MultirootFSolverCache, x0::AbstractArray)
    for (i, v) in enumerate(x0)
        v = convert(Cdouble, v)
        GSL.vector_set(ca.vinit, i-1, v)
    end
end

function _test_delta(ca::GSL_MultirootFSolverCache, xtol)
    x = GSL.multiroot_fsolver_root(ca.solver)
    dx = GSL.multiroot_fsolver_dx(ca.solver)
    # Only consider absolute tolerance
    return GSL.multiroot_test_delta(dx, x, xtol, 0)
end

function _test_residual(ca::GSL_MultirootFSolverCache, ftol)
    f = GSL.multiroot_fsolver_f(ca.solver)
    return GSL.multiroot_test_residual(f, ftol)
end

function _solve!(ca::GSL_MultirootFSolverCache, xtol, ftol, maxiter, verbose, pgap)
    iter = 0
    status, xconverged, fconverged = -2, -2, -2
    converged = false
    success = GSL.GSL_SUCCESS
    while iter < maxiter
        iter += 1
        verbose && iszero(iter%pgap) && println("  iteration $iter...")
        status = GSL.multiroot_fsolver_iterate(ca.solver)
        xtol === Inf || (xconverged = _test_delta(ca, xtol))
        ftol === Inf || (fconverged = _test_residual(ca, ftol))
        # Convergence requires only one of the two criteria
        converged = status==success && (xconverged==success || fconverged==success)
        converged && break
    end
    converged || @warn "solver did not converge with xtol=$xtol and ftol=$ftol after $iter iterations"
    converged && verbose && println("solver converged with xtol=$xtol and ftol=$ftol after $iter iterations")
    x = GSL.multiroot_fsolver_root(ca.solver)
    # Copy the solution
    sol = Vector{Float64}(unsafe_wrap(Array{Cdouble}, GSL.vector_ptr(x, 0), ca.N))
    return sol, converged
end

function solve!(ca::GSL_MultirootFSolverCache, f!, x0::AbstractArray;
        xtol::Real=1e-8, ftol::Real=Inf, maxiter::Int=1000, verbose=false, kwargs...)
    ca.N == length(x0) || throw(DimensionMismatch(
        "the length of x0 does not match the solver cache"))
    verbose, pgap = _verbose(verbose)
    _setvinit!(ca, x0)
    function _f(x_vec, p, y_vec)
        x = GSL.wrap_gsl_vector(x_vec)
        y = GSL.wrap_gsl_vector(y_vec)
        # Switch the order of arguments
        f!(y, x)
        return Cint(GSL.GSL_SUCCESS)
    end
    cfunc = @cfunction($_f, Cint, (Ptr{GSL.gsl_vector}, Ptr{Cvoid}, Ptr{GSL.gsl_vector}))
    GC.@preserve cfunc begin
        gsl_func = GSL.gsl_multiroot_function(Base.unsafe_convert(Ptr{Cvoid}, cfunc), ca.N, 0)
        GSL.multiroot_fsolver_set(ca.solver, gsl_func, ca.vinit)
        sol, converged = _solve!(ca, xtol, ftol, maxiter, verbose, pgap)
    end
    return sol, converged
end

function solve!(t::Type{<:GSL_MultirootFSolver}, f!, x0::AbstractArray; kwargs...)
    ca = GSL_MultirootFSolverCache(t, length(x0))
    res = solve!(ca, f!, x0; kwargs...)
    GSL.multiroot_fsolver_free(ca.solver)
    GSL.vector_free(ca.vinit)
    return res
end
