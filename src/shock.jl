struct ShockProcess{σ,ins,outs,F<:Function} <: AbstractBlock{ins,outs}
    f!::F
    function ShockProcess(σ::Symbol, ins::NTuple{NI,Symbol}, outs::NTuple{NO,Symbol},
            f!::F) where {NI,NO,F}
        NO == 1 || throw(ArgumentError("ShockProcess can only have one output variable"))
        return new{σ,ins,outs,F}(f!)
    end
end

shockse(::ShockProcess{σ}) where σ = σ

function impulse!(out::AbstractVecOrMat, sh::ShockProcess, paravals::NamedTuple)
    invals = NamedTuple{inputs(sh)}(paravals)
    if all(x->x isa Real, invals)
        sh.f!(out, invals...)
        return out
    else
        # Each parameter should have the same length
        N = length(invals[1])
        outb = _block1(out, Int(size(out,1)/N))
        for i in 1:N
            sh.f!(view(outb, Block(i,1)), map(Fix2(getindex, i), invals)...)
        end
        return out
    end
end

ar1impulse!(out, ar::Real) = impulse!(out, ARMAProcess(ar, ()))
ar1shock(σ::Symbol, ar::Symbol, out::Symbol) = ShockProcess(σ, (ar,), (out,), ar1impulse!)
arma11impulse!(out, ar::Real, ma::Real) = impulse!(out, ARMAProcess(ar, ma))
arma11shock(σ::Symbol, ar::Symbol, ma::Symbol, out::Symbol) =
    ShockProcess(σ, (ar, ma), (out,), arma11impulse!)

function impulse!(out::AbstractVecOrMat, gs::GMaps, exovar::Symbol, endovar::Symbol,
        shockirf::AbstractVecOrMat)
    # For array variable, shocks are arranged by column
    # Number of rows should be the number of time horizons multiplied by the variable width
    N = size(out, 2)
    size(shockirf, 2) == N || throw(DimensionMismatch(
        "out and shockirf are expected to have the same number of columns"))
    if N == 1
        G = gs(exovar, endovar)
        return mul!(out, G, shockirf)
    else
        wendo = length(gs.gj.tjac.varvals[][endovar])
        nT = Int(size(out,1)/wendo)
        nT == gs.gj.nTfull || throw(DimensionMismatch(
            "out is expected to fit all horizons available from gs"))
        G = gs(exovar, endovar)
        for n in 1:N
            Gn = view(G,:,1+(n-1)*nT:n*nT)
            mul!(view(out,:,n), Gn, view(shockirf,:,n))
        end
        return out
    end
end

function impulse(gs::GMaps, exovar::Symbol, endovar::Symbol, shockirf::AbstractVecOrMat)
    nT = gs.gj.nTfull
    wendo = length(gs.gj.tjac.varvals[][endovar])
    out = similar(shockirf, wendo*nT, size(shockirf,2))
    return impulse!(out, gs, exovar, endovar, shockirf)
end

function impulse(gs::GMaps{TF}, dZs::ValidVarInput, endovars=nothing;
        transform=false) where TF
    gj = gs.gj
    tjac = gj.tjac
    dZs isa Pair && (dZs = (dZs,))
    endovars isa Symbol && (endovars = (endovars,))
    endovars === nothing && (endovars = setdiff(tjac.pool, gj.exovars, tjac.tars))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    nT = gj.nTfull
    out = Dict{Symbol, Dict{Symbol,Array{TF,3}}}()
    for (exo, dZ) in dZs
        size(dZ,1) == nT || throw(ArgumentError("the length of shock $exo is not $nT"))
        d = Dict{Symbol, Array{TF,3}}()
        out[exo] = d
        for endo in endovars
            irf = impulse(gs, exo, endo, dZ)
            # Always use three-dimensional array to avoid ambiguity due to array variables
            # Dimensions: horizons, endo vars, exo vars
            irf = reshape(irf, nT, size(irf,1)÷nT, size(irf,2))
            d[endo] = irf
        end
    end
    _transform!(out, transform, gj.tjac.varvals[])
    return out
end

# ! Can easily result in too many columns with array variables
function aswidetable(irfs::Dict{Symbol})
    out = Pair[]
    printed = false
    for (s, d) in irfs
        for (k, v) in d
            _, M, N = size(v)
            for j in axes(v, 3)
                for i in axes(v, 2)
                    if M == N == 1
                        name = Symbol(s, '_', k)
                    elseif M == 1
                        name = Symbol(s, j, '_', k)
                    elseif N == 1
                        name = Symbol(s, '_', k, i)
                    else
                        printed || @info "Consider aslongtable instead of aswidetable for performance"
                        printed = true
                        name = Symbol(s, j, '_', k, i)
                    end
                    push!(out, name=>v[:,i,j])
                end
            end
        end
    end
    return (; out...)
end

function aslongtable(irfs::Dict{Symbol})
    exos = Symbol[]
    endos = Symbol[]
    vals = Float64[]
    for (s, d) in irfs
        for (k, v) in d
            _, M, N = size(v)
            L = length(v)
            L0 = length(vals)
            resize!(exos, L0+L)
            resize!(endos, L0+L)
            resize!(vals, L0+L)
            r = L0
            # Symbol allocates and hence better to be created once before the loop
            exonames = N == 1 ? (s,) : ntuple(j->Symbol(s, j), N)
            endonames = M == 1 ? (k,) : ntuple(i->Symbol(k, i), M)
            @inbounds for j in axes(v, 3)
                exoname = exonames[j]
                for i in axes(v, 2)
                    endoname = endonames[i]
                    for t in axes(v, 1)
                        r += 1
                        exos[r] = exoname
                        endos[r] = endoname
                        vals[r] = convert(Float64, v[t,i,j])
                    end
                end
            end
        end
    end
    return (exovar=exos, endovar=endos, value=vals)
end

function _simul_shock!(dY::AbstractVecOrMat, dX::AbstractVecOrMat, ε::AbstractVecOrMat, nT::Int)
    T = min(length(dY), size(ε,1)-nT+1)
    Nout = size(dY, 2)
    for n in 1:Nout
        for t in 1:T
            # The order of ε is flipped
            dY[t,n] = dot(view(dX,1+(n-1)*nT:n*nT,:), view(ε,t+nT-1:-1:t,:))
        end
    end
    return dY
end

_addssval!(out::VecOrMat, v::Real, T::Int) = (out .+= v)

function _addssval!(out::VecOrMat, vs::AbstractArray, T::Int)
    for (n, v) in enumerate(vs)
        for t in 1:T
            out[t,n] += v
        end
    end
end

function simulate!(out::AbstractVecOrMat, gs::GMaps, exovar::Symbol, endovar::Symbol,
        dX::AbstractVecOrMat, ε::AbstractVecOrMat, shockirf::AbstractVecOrMat;
        addssval::Bool=true)
    gj = gs.gj
    nT = gj.nTfull
    impulse!(dX, gs, exovar, endovar, shockirf)
    _simul_shock!(out, dX, ε, nT)
    if addssval
        T = min(length(out), size(ε,1)-nT+1)
        _addssval!(out, gj.tjac.varvals[][endovar], T)
    end
    return out
end

function simulate(gs::GMaps{T1}, exovar::Symbol, endovar::Symbol,
        ε::AbstractVecOrMat{T2}, shocks; kwargs...) where {T1,T2}
    gj = gs.gj
    Nout = length(gj.tjac.varvals[][endovar])
    Nin = size(ε, 2)
    T = size(ε,1) - gj.nTfull + 1
    T < 1 && throw(ArgumentError("length of shocks is smaller than $(gj.nTfull)"))
    TF = promote_type(T1,T2)
    out = Matrix{TF}(undef, T, Nout)
    dX = Matrix{TF}(undef, Nout*gj.nTfull, Nin)
    if shocks isa AbstractVecOrMat{<:Real}
        shockirf = shocks
    else
        shocks isa Union{Tuple, AbstractVector} || (shocks = (shocks,))
        Nsh = length(shocks)
        Nsh == size(ε,2) || throw(DimensionMismatch(
            "number of shocks is expected to be $(size(ε,2))"))
        shockirf = Matrix{TF}(undef, gj.nTfull, Nsh)
        for i in 1:Nsh
            impulse!(view(shockirf,:,i), shocks[i])
        end
    end
    simulate!(out, gs, exovar, endovar, dX, ε, shockirf; kwargs...)
    return out
end
