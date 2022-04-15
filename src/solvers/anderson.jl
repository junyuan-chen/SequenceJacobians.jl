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
