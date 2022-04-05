const ValidPathInput{TF<:AbstractFloat} = Union{Pair{Symbol,<:AbstractVecOrMat{TF}},
    Vector{<:Pair{Symbol,<:AbstractVecOrMat{TF}}},
    Dict{Symbol,<:AbstractVecOrMat{TF}}}
const PathType{TF<:AbstractFloat} = Union{TF,Pair{Int,Vector{TF}},Pair{Int,Matrix{TF}}}

struct Transition{TF<:AbstractFloat}
    tjac::TotalJacobian{TF}
    exovars::Vector{Symbol}
    unknowns::Vector{Symbol}
    varpaths::Dict{Symbol,PathType{TF}}
    H_U::Union{Matrix{TF},LU{TF,Matrix{TF}}}
    upaths::Vector{TF}
    resids::Vector{TF}
end

function Transition(tjac::TotalJacobian{TF}, exopaths::ValidPathInput,
        initials::Union{ValidPathInput,Nothing}=nothing;
        H_U::Union{LU,Nothing}=nothing, solveH_U::Bool=true) where TF
    nT = tjac.nT
    varvals = tjac.varvals
    exopaths isa Pair && (exopaths = (exopaths,))
    initials !== nothing && initials isa Pair && (initials = (initials,))
    varpaths = Dict{Symbol,PathType{TF}}()
    if initials !== nothing
        for (k, v) in initials
            varpaths[k] = v isa Pair ? v : 0=>v
        end
    end
    exovars = Symbol[]
    for (k, v) in exopaths
        k in tjac.srcs || throw(ArgumentError("$k is not a source variable"))
        push!(exovars, k)
        varpaths[k] = v isa Pair ? v : 0=>v
    end
    upaths = TF[]
    unknowns = collect(setdiff(tjac.srcs, exovars))
    for u in unknowns
        upath = get(varpaths, u, nothing)
        if upath === nothing
            upath = fill(varvals[u], nT)
            varpaths[u] = 0=>upath
            append!(upaths, upath)
        else
            upath isa Pair && (upath = upath[2])
            if upath isa Matrix
                throw(ArgumentError("only scalar variable is supported for the unknowns"))
            elseif upath isa Vector
                l = length(upath)
                if l < nT
                    resize!(upath, nT)
                    upath[l+1:end] .= upath[l]
                end
                append!(upaths, upath)
            end
        end
    end
    nU = length(unknowns)
    ntar = length(tjac.tars)
    if H_U === nothing
        zmap = LinearMap(UniformScaling(zero(TF)), nT)
        H_U = Matrix(hvcat(((nU for _ in 1:ntar)...,),
            (get(tjac.totals[v], t, zmap) for t in tjac.tars for v in unknowns)...))
        solveH_U && (H_U = lu!(H_U))
    end
    for blk in tjac.blks
        # Any output must have not been met before unless provided by initials
        for ov in outputs(blk)
            outpath = get(varpaths, ov, nothing)
            if outpath === nothing
                varpaths[ov] = 0=>fill(varvals[ov], nT)
            end
        end
        for inv in invars(blk)
            n = name(inv)
            inpath = get(varpaths, n, nothing)
            if inpath === nothing
                varpaths[n] = varvals[n]
            elseif inpath isa Pair
                s = shift(inv)
                if !iszero(s)
                    i0 = inpath[1]
                    inpath = inpath[2]
                    if s < 0
                        if i0 + s >= 0
                            continue
                        else
                            pushfirst!(inpath, (inpath[1] for i in 1:-i0-s)...)
                            varpaths[n] = -s=>inpath
                        end
                    else
                        ilast = i0 + s + nT
                        if ilast <= length(inpath)
                            continue
                        else
                            push!(inpath, (inpath[end] for i in length(inpath)+1:ilast)...)
                        end
                    end
                end
            end
        end
    end
    resids = Vector{TF}(undef, nT*ntar)
    return Transition(tjac, exovars, unknowns, varpaths, H_U, upaths, resids)
end

function _inputs!(tr::Transition, inputs::AbstractVector)
    nT = tr.tjac.nT
    i0 = 0
    @inbounds for u in tr.unknowns
        path = tr.varpaths[u]
        if path isa Pair
            s, path = path
        else
            s = 0
        end
        for i in 1:nT
            path[s+i] = inputs[i0+i]
        end
        i0 += nT
    end
end

function _resids!(resids::AbstractVector, tr::Transition)
    nT = tr.tjac.nT
    varvals = tr.tjac.varvals
    i0 = 0
    @inbounds for tar in tr.tjac.tars
        path = tr.varpaths[tar]
        if path isa Pair
            s, path = (path[1], path[2])
        else
            s = 0
        end
        # It is important to subtract the steady-state residuals of the targets
        val = varvals[tar]
        for i in 1:nT
            resids[i0+i] = path[s+i] - val
        end
        i0 += nT
    end
end

function residuals!(resids::AbstractVector, tr::Transition)
    for b in tr.tjac.blks
        transition!(tr.varpaths, b, tr.tjac.nT)
    end
    _resids!(resids, tr)
    return resids
end

residuals!(tr::Transition) = residuals!(tr.resids, tr)

function residuals!(resids::AbstractVector, tr::Transition, inputs::AbstractVector)
    _inputs!(tr, inputs)
    return residuals!(resids, tr)
end

residuals!(tr::Transition, inputs::AbstractVector) =
    residuals!(tr.resids, tr, inputs)

function update!(tr::Transition)
    residuals!(tr, tr.upaths)
    tr.upaths .-= ldiv!(tr.H_U, tr.resids)
    return tr.resids
end

function solve!(tr::Transition; tol::Real=1e-8, maxit::Int=50, pnorm::Real=Inf,
        verbose::Bool=true)
    count = 0
    diffnorm = Inf
    for i in 1:maxit
        count += 1
        diffnorm = norm(update!(tr), pnorm)
        verbose && println("  Iteration: $i   $pnorm-norm: $diffnorm")
        diffnorm < tol && break
    end
    if diffnorm < tol
        verbose && println("Converged in $count iterations!")
    else
        @warn "Convergence failed with $pnorm-norm being $diffnorm"
    end
end
