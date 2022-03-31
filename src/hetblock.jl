struct HetBlock{HA<:AbstractHetAgent} <: AbstractBlock
    ins::Vector{Symbol}
    invars::Vector{VarSpec}
    ssins::Set{Symbol}
    outs::Vector{Symbol}
    ha::HA
    ssargs::Dict{Symbol,Any}
    jacargs::Dict{Symbol,Any}
    function HetBlock(ins::Vector{Symbol}, invars::Vector{VarSpec}, ssins::Set{Symbol},
            outs::Vector{Symbol}, ha::HA, ssargs::Dict{Symbol,Any},
            jacargs::Dict{Symbol,Any}) where HA
        _checkinsouts(ins, outs, ssins)
        return new{HA}(ins, invars, ssins, outs, ha, ssargs, jacargs)
    end
end

function block(ha::AbstractHetAgent, ins, outs; ssins=ins, ssargs=nothing, jacargs=nothing)
    ssargs = ssargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(ssargs...)
    jacargs = jacargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(jacargs...)
    return HetBlock(_inout(ins, outs, ssins)..., ha, ssargs, jacargs)
end

_verbose(verbose::Integer) = (verbose>0, Int(verbose))
_verbose(verbose::Bool) = (verbose, 20)
_verbose(verbose) = throw(ArgumentError("invalid specification of keyword verbose"))

function _backwardss!(ha, invals, maxbackiter, backtol, verbose, pgap)
    iter = 0
    backconverged = false
    while iter < maxbackiter
        iter += 1
        verbose && iszero(iter%pgap) && println("  backward iteration $iter...")
        backward!(ha, invals...)
        st = backward_status(ha)
        st === nothing || verbose && println("    ", st)
        backconverged = backward_converged(ha, st, backtol)
        backconverged && break
    end
    backconverged ||
        @warn "backward iterations did not converge with tolerance $backtol after $iter steps"
    backconverged && verbose &&
        println("backward iteration converged with tolerance $backtol after $iter steps")
    return backconverged
end

function _forwardss!(ha, invals, maxforiter, fortol, verbose, pgap)
    forconverged = false
    iter = 0
    while iter < maxforiter
        iter += 1
        verbose && iszero(iter%pgap) && println("  forward iteration $iter...")
        forward!(ha, invals...)
        st = forward_status(ha)
        st === nothing || verbose && println("    ", st)
        forconverged = forward_converged(ha, st, fortol)
        forconverged && break
    end
    forconverged ||
        @warn "forward iterations did not converge with tolerance $fortol after $iter steps"
    forconverged && verbose &&
        println("forward iteration converged with tolerance $fortol after $iter steps")
    return forconverged
end

function steadystate!(b::HetBlock, varvals::AbstractDict)
    ha = b.ha
    ins = inputs(b)
    invals = ntuple(i->varvals[ins[i]], length(ins))
    verbose, pgap = _verbose(get(b.ssargs, :verbose, false))
    # Backward iterations
    backward_init!(ha, invals...)
    maxbackiter = get(b.ssargs, :maxbackiter, 1000)
    backtol = get(b.ssargs, :backtol, 1e-8)
    backconverged = _backwardss!(ha, invals, maxbackiter, backtol, verbose, pgap)
    # Forward iterations
    forward_init!(ha, invals...)
    maxforiter = get(b.ssargs, :maxforiter, 1000)
    fortol = get(b.ssargs, :fortol, 1e-10)
    forconverged = _forwardss!(ha, invals, maxforiter, fortol, verbose, pgap)
    backconverged && forconverged && (b.jacargs[:ssfound] = true)
    # Obtain aggregate outcomes
    vals = aggregate(ha, invals...)
    for (i, n) in enumerate(outputs(b))
        val = get(varvals, n, nothing)
        val isa AbstractArray ? copyto!(val, vals[i]) : (varvals[n] = vals[i])
    end
end

struct HetAgentJacCache{HA<:AbstractHetAgent, FX<:Tuple, DC, TF<:AbstractFloat, M, N}
    ha::HA
    hass::HA
    nT::Int
    fxs::FX
    df::Array{TF,M}
    dcache::DC
    dEVs::Vector{Array{TF,N}}
    dYs::Dict{Int,Matrix{TF}}
    dDs::Dict{Int,Array{TF,M}}
    Es::Dict{Int,Array{TF,M}}
    Js::Dict{Int,Dict{Int,Matrix{TF}}}
end

function _expectation_vector!(E, Etemp, nT, pol, exogs, endos)
    vEs = splitdimsview(E)
    backward!(vEs[1], pol, exogs...)
    vEs[1] .-= mean(vEs[1])
    @inbounds for t in 2:nT-1
        backward!(Etemp, vEs[t-1], endos...)
        backward!(vEs[t], Etemp, exogs...)
        vEs[t] .-= mean(vEs[t])
    end
end

function HetAgentJacCache(b::HetBlock, nT::Int)
    # Check whether steady state has been found
    get(b.jacargs, :ssfound, false) ||
        error("must find the steady state for HetBlock before computing Jacobians")
    ha = deepcopy(b.ha)
    hass = b.ha
    D = getdist(ha)
    TF = eltype(D)
    ssize = size(D)
    N = ndims(D)
    M = N + 1
    fxs = (getvalues(ha)..., getpolicies(ha)...)
    df = Array{TF,M}(undef, ssize..., length(fxs))
    dcache = GradientCache(df, one(TF))

    dEVs = [Array{TF,N}(undef, ssize...) for _ in 1:length(valuevars(ha))]
    dYs = Dict{Int,Matrix{TF}}()
    dDs = Dict{Int,Array{TF,M}}()
    Es = Dict{Int,Array{TF,M}}()
    exogs = exogprocs(ha)
    endos = endoprocs(ha)
    if nT > 1
        Etemp = similar(D)
        for (i, n) in enumerate(policies(ha))
            E = Array{TF,M}(undef, ssize..., nT-1)
            Es[i] = E
            pol = getpolicy(ha, n)
            _expectation_vector!(E, Etemp, nT, pol, exogs, endos)
        end
    end
    Js = Dict{Int,Dict{Int,Matrix{TF}}}()
    return HetAgentJacCache{typeof(ha),typeof(fxs),typeof(dcache),TF,M,N}(
        ha, hass, nT, fxs, df, dcache, dEVs, dYs, dDs, Es, Js)
end

function _setEV!(ev::AbstractArray, evss::AbstractArray, dev::AbstractArray, x::Real)
    @simd for i in eachindex(ev)
        @inbounds ev[i] = evss[i] + x * dev[i]
    end
end

function _setJ!(ca::HetAgentJacCache, i::Int, npol::Int)
    nT = ca.nT
    Ji = get!(valtype(ca.Js), ca.Js, i)
    dD = ca.dDs[i]
    N = Int(length(dD)/nT)
    dD = reshape(dD, N, nT)
    _Jio() = valtype(Ji)(undef, nT, nT)
    @inbounds for o in 1:npol
        Jio = get!(_Jio, Ji, o)
        copyto!(view(Jio, 1, :), view(ca.dYs[i],:,o))
        if nT > 1
            E = reshape(ca.Es[o], N, nT-1)
            mul!(view(Jio, 2:nT, :), E', dD)
            for s in 2:nT
                for t in 2:nT
                    Jio[t,s] += Jio[t-1,s-1]
                end
            end
        end
    end
end

function _jacobian!(ca::HetAgentJacCache, i::Int, nT::Int, varvals, ins, val, evs, evsss)
    ha = ca.ha
    hass = ca.hass

    function f1!(fx, x)
        vfxs = splitdimsview(fx)
        xs = (ntuple(k->varvals[ins[k]], i-1)..., x,
            ntuple(k->varvals[ins[k+i]], length(ins)-i)...)
        backward_endo!(ha, expectedvalues(ha)..., xs...)
        for k in eachindex(ca.fxs)
            @inbounds copyto!(vfxs[k], ca.fxs[k])
        end
    end

    finite_difference_gradient!(ca.df, f1!, val, ca.dcache)

    exogs = exogprocs(ha)
    nV = length(valuevars(ha))
    Vs = splitdimsview(ca.df)
    for k in 1:nV
        backward!(ca.dEVs[k], Vs[k], exogs...)
    end

    Dss = getdist(hass)
    TF = eltype(Dss)
    ssize = size(Dss)
    _dD() = valtype(ca.dDs)(undef, ssize..., nT)
    dD = get!(_dD, ca.dDs, i)
    dDs = splitdimsview(dD)

    # Assume policies associated with endogenous states are placed in the front
    das = ntuple(k->Vs[k+nV], length(endostates(ha)))
    forward_shock!(dDs[1], Dss, endoprocs(ha)..., das...)

    npol = length(policies(ha))
    _dY() = Matrix{TF}(undef, nT, npol)
    dY = get!(_dY, ca.dYs, i)
    vDss = view(Dss, :)
    vas = ntuple(k->view(Vs[k+nV],:), npol)
    strvDss = stride1(vDss)
    @inbounds for o in eachindex(vas)
        va = vas[o]
        dY[1,o] = BLAS.dot(length(vDss), vDss, strvDss, va, stride1(va))
    end

    xs = ntuple(k->varvals[ins[k]], length(ins))
    function f!(fx, x)
        vfxs = splitdimsview(fx)
        @inbounds for k in 1:nV
            _setEV!(evs[k], evsss[k], ca.dEVs[k], x)
        end
        backward_endo!(ha, evs..., xs...)
        @inbounds for k in eachindex(ca.fxs)
            copyto!(vfxs[k], ca.fxs[k])
        end
    end

    if nT > 1
        z = zero(TF)
        for t in 2:nT
            finite_difference_gradient!(ca.df, f!, z, ca.dcache)
            for k in 1:nV
                backward!(ca.dEVs[k], Vs[k], exogs...)
            end
            forward_shock!(dDs[t], Dss, endoprocs(ha)..., das...)
            @inbounds for o in eachindex(vas)
                va = vas[o]
                dY[t,o] = BLAS.dot(length(vDss), vDss, strvDss, va, stride1(va))
            end
        end
    end

    _setJ!(ca, i, npol)
end

# ! To do: consider shocks to exogenous law of motion and aggregation method
function jacobian(b::HetBlock, i::Int, nT::Int, varvals::Dict{Symbol,<:ValType})
    _makejacca() = HetAgentJacCache(b, nT)
    ca = get!(_makejacca, b.jacargs, :jacca)
    # Still need to allocate a new one if the existing one is not usable
    ca.nT == nT && ca.hass === b.ha || (ca = HetAgentJacCache(b, nT))

    ins = inputs(b)
    vi = ins[i]
    val = varvals[vi]
    evs = expectedvalues(ca.ha)
    evsss = expectedvalues(ca.hass)

    _jacobian!(ca, i, nT, varvals, ins, val, evs, evsss)

    return ca
end

function getjacmap(b::HetBlock, J::HetAgentJacCache,
        i::Int, ii::Int, r::Int, rr::Int, nT::Int)
    j = J.Js[i][r]
    return LinearMap(j), false
end
