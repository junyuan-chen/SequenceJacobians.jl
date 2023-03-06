struct ShockProcess{σ,ins,outs,F<:Function} <: AbstractBlock{ins,outs}
    f!::F
    function ShockProcess(σ::Symbol, ins::NTuple{NI,Symbol}, outs::NTuple{NO,Symbol},
            f!::F) where {NI,NO,F}
        NO == 1 || throw(ArgumentError("ShockProcess can only have one output variable"))
        return new{σ,ins,outs,F}(f!)
    end
end

shockse(::ShockProcess{σ}) where σ = σ

impulse!(out, sh::ShockProcess, paravals::NamedTuple) =
    sh.f!(out, map(k->getfield(paravals, k), inputs(sh))...)

_ar1!(out, ar::Real) = impulse!(out, ARMAProcess(ar, ()))
ar1shock(σ::Symbol, ar::Symbol, out::Symbol) = ShockProcess(σ, (ar,), (out,), _ar1!)
_arma11!(out, ar::Real, ma::Real) = impulse!(out, ARMAProcess(ar, ma))
arma11shock(σ::Symbol, ar::Symbol, ma::Symbol, out::Symbol) =
    ShockProcess(σ, (ar, ma), (out,), _arma11!)

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
    G = gs[exovar, endovar]
    N = size(ε, 2)
    if N == 1
        mul!(dX, G, shockirf)
    else
        for n in 1:N
            Gn = view(G,:,1+(n-1)*nT:n*nT)
            mul!(view(dX,:,n), Gn, view(shockirf,:,n))
        end
    end
    _simul_shock!(out, dX, ε, nT)
    if addssval
        T = min(length(out), size(ε,1)-nT+1)
        _addssval!(out, gj.tjac.varvals[endovar], T)
    end
    return out
end

function simulate(gs::GMaps{T1}, exovar::Symbol, endovar::Symbol,
        ε::AbstractVecOrMat{T2}, shocks; kwargs...) where {T1,T2}
    gj = gs.gj
    Nout = length(gj.tjac.varvals[endovar])
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
