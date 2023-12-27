struct HetBlock{HA<:AbstractHetAgent,TF<:AbstractFloat,ins,outs,NI} <: AbstractBlock{ins,outs}
    invars::NTuple{NI,VarSpec}
    ssins::Set{Symbol}
    ha::HA
    ssargs::Dict{Symbol,Any}
    ssstatus::Ref{NamedTuple{(:initialized, :solved), Tuple{Bool, Bool}}}
    jacargs::Dict{Symbol,Any}
    diffargs::Ref{NamedTuple{(:twosided, :epsilon), Tuple{Bool, TF}}}
    function HetBlock(ins::NTuple{NI,Symbol}, invars::NTuple{NI,VarSpec},
            ssins::Set{Symbol}, outs::NTuple{NO,Symbol}, ha::HA, ssargs::Dict{Symbol,Any},
            jacargs::Dict{Symbol,Any}) where {NI,NO,HA}
        _checkinsouts(ins, outs, ssins)
        ssstatus = Ref((initialized=false, solved=false))
        TF = eltype(getdist(ha))
        diffargs = Ref((twosided=true, epsilon=default_relstep(Val(:central), TF)))
        return new{HA,TF,ins,outs,NI}(invars, ssins, ha, ssargs, ssstatus, jacargs, diffargs)
    end
end

function block(ha::AbstractHetAgent, ins, outs; ssins=ins, ssargs=nothing, jacargs=nothing)
    ssargs = ssargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(ssargs...)
    jacargs = jacargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(jacargs...)
    return HetBlock(_inout(ins, outs, ssins)..., ha, ssargs, jacargs)
end

_verbose(verbose::Integer) = (verbose>0, Int(verbose))
_verbose(verbose::Bool) = (verbose, 100)

function _backward!(ha, invals, maxbackiter, backtol, verbose, pgap)
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

function _forward!(ha, invals, maxforiter, fortol, verbose, pgap)
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

function backward!(::Nothing, b::HetBlock, invals::Tuple)
    verbose, pgap = _verbose(get(b.ssargs, :verbose, false))
    maxbackiter = get(b.ssargs, :maxbackiter, 5000)
    backtol = get(b.ssargs, :backtol, 1e-8)
    return _backward!(b.ha, invals, maxbackiter, backtol, verbose, pgap)
end

function forward!(::Nothing, b::HetBlock, invals::Tuple)
    initendo!(b.ha)
    verbose, pgap = _verbose(get(b.ssargs, :verbose, false))
    maxforiter = get(b.ssargs, :maxforiter, 5000)
    fortol = get(b.ssargs, :fortol, 1e-10)
    return _forward!(b.ha, invals, maxforiter, fortol, verbose, pgap)
end

function steadystate!(b::HetBlock, varvals::NamedTuple)
    invals = map(k->getfield(varvals, k), inputs(b))
    initialized = b.ssstatus[].initialized
    if !initialized
        backward_init!(b.ha, invals...)
        initdist!(b.ha)
        b.ssstatus[] = merge(b.ssstatus[], (initialized=true,))
    end
    bsolver = backwardsolver(b.ha)
    backconverged = backward!(bsolver, b, invals)
    fsolver = forwardsolver(b.ha)
    forconverged = forward!(fsolver, b, invals)
    backconverged && forconverged && (b.ssstatus[] = merge(b.ssstatus[], (solved=true,)))
    # Obtain aggregate outcomes
    vals = aggregate(b.ha, invals...)
    return merge(varvals, NamedTuple{outputs(b)}(vals))
end

struct HetAgentJacCache{HA<:AbstractHetAgent, FX<:Tuple, DC, TF<:AbstractFloat, M, N}
    ha::HA
    hass::HA
    nT::Int
    epsilon::TF
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
    b.ssstatus[].solved ||
        error("must find the steady state for HetBlock before computing Jacobians")
    ha = deepcopy(b.ha)
    hass = b.ha
    D = getdist(ha)
    TF = eltype(D)
    ssize = size(D)
    N = ndims(D)
    M = N + 1
    Vs = valuevars(ha)
    pols = policies(ha)
    fxs = (Vs..., pols...)
    df = Array{TF,M}(undef, ssize..., length(fxs))
    fdtype = b.diffargs[].twosided ? Val(:central) : Val(:forward)
    epsilon = b.diffargs[].epsilon
    dcache = GradientCache(df, one(TF), fdtype)

    dEVs = [Array{TF,N}(undef, ssize...) for _ in 1:length(Vs)]
    dYs = Dict{Int,Matrix{TF}}()
    dDs = Dict{Int,Array{TF,M}}()
    Es = Dict{Int,Array{TF,M}}()
    exogs = exogprocs(ha)
    endos = endoprocs(ha)
    if nT > 1
        Etemp = similar(D)
        for (i, n) in enumerate(pols)
            E = Array{TF,M}(undef, ssize..., nT-1)
            Es[i] = E
            pol = pols[i]
            _expectation_vector!(E, Etemp, nT, pol, exogs, endos)
        end
    end
    Js = Dict{Int,Dict{Int,Matrix{TF}}}()
    return HetAgentJacCache{typeof(ha),typeof(fxs),typeof(dcache),TF,M,N}(
        ha, hass, nT, epsilon, fxs, df, dcache, dEVs, dYs, dDs, Es, Js)
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

function _jacobian!(b::HetBlock, ca::HetAgentJacCache, i::Int, nT::Int, varvals, evs, evsss)

    ins = inputs(b)
    ha = ca.ha
    hass = ca.hass

    function f1!(fx, x)
        vfxs = splitdimsview(fx)
        xs = (map(k->getfield(varvals, k), ins[1:i-1])..., x,
            map(k->getfield(varvals, k), ins[i+1:length(ins)])...)
        backward_endo!(ha, expectedvalues(ha)..., xs...)
        for k in eachindex(ca.fxs)
            @inbounds copyto!(vfxs[k], ca.fxs[k])
        end
    end

    val = varvals[ins[i]]
    finite_difference_gradient!(ca.df, f1!, val, ca.dcache, absstep=ca.epsilon)

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
    endos = endoprocs(ha)
    das = ntuple(k->Vs[k+nV], length(endos))
    forward_shock!(dDs[1], Dss, endos..., das...)

    npol = length(policies(ha))
    dY = haskey(ca.dYs, i) ? ca.dYs[i] : (ca.dYs[i] = Matrix{TF}(undef, nT, npol))
    vDss = view(Dss, :)
    vas = ntuple(k->view(Vs[k+nV],:), npol)
    strvDss = stride1(vDss)
    @inbounds for o in eachindex(vas)
        va = vas[o]
        dY[1,o] = BLAS.dot(length(vDss), vDss, strvDss, va, stride1(va))
    end

    xsss = map(k->getfield(varvals, k), inputs(b))
    function f!(fx, x)
        vfxs = splitdimsview(fx)
        @inbounds for k in 1:nV
            _setEV!(evs[k], evsss[k], ca.dEVs[k], x)
        end
        backward_endo!(ha, evs..., xsss...)
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
            forward_shock!(dDs[t], Dss, endos..., das...)
            @inbounds for o in eachindex(vas)
                va = vas[o]
                dY[t,o] = BLAS.dot(length(vDss), vDss, strvDss, va, stride1(va))
            end
        end
    end

    _setJ!(ca, i, npol)
end

_fdtype(::GradientCache{T1,T2,T3,T4,fdtype}) where {T1,T2,T3,T4,fdtype} = fdtype

function _getjaccache(b::HetBlock, nT::Int)
    _makejacca() = HetAgentJacCache(b, nT)
    if haskey(b.jacargs, :jacca)
        ca = b.jacargs[:jacca]
        ca.nT == nT && ca.hass === b.ha || return _makejacca()
        twosided = b.diffargs[].twosided
        if twosided
            _fdtype(ca.dcache) == Val(:central) || return _makejacca()
        else
            _fdtype(ca.dcache) == Val(:forward) || return _makejacca()
        end
        b.diffargs[].epsilon == ca.epsilon || return _makejacca()
        return ca
    else
        return _makejacca()
    end
end

struct HetBlockJacobian{BLK<:HetBlock, TF, CA<:HetAgentJacCache} <: MatrixBlockJacobian{TF}
    blk::BLK
    ca::CA
    iins::Vector{Int}
    nT::Int
end

function (j::HetBlockJacobian)(varvals::NamedTuple)
    evs = expectedvalues(j.ca.ha)
    evsss = expectedvalues(j.ca.hass)
    for i in j.iins
        _jacobian!(j.blk, j.ca, i, j.nT, varvals, evs, evsss)
    end
    return j
end

# ! To do: consider shocks to exogenous law of motion and aggregation method
function jacobian(b::HetBlock{HA,TF}, iins, nT::Int, varvals::NamedTuple) where {HA,TF}
    ca = _getjaccache(b, nT)
    b.jacargs[:jacca] = ca
    j = HetBlockJacobian{typeof(b),TF,typeof(ca)}(b, ca, collect(iins), nT)
    return j(varvals)
end

@inline getindex(j::HetBlockJacobian, r::Int, i::Int) = j.ca.Js[j.iins[i]][r]

show(io::IO, b::HetBlock) = print(io, "HetBlock($(b.ha))")

function show(io::IO, ::MIME"text/plain", b::HetBlock)
    println(io, "HetBlock($(b.ha)):")
    _showinouts(io, b)
end

function show(io::IO, j::HetBlockJacobian)
    print(io, "HetBlockJacobian(", j.blk.ha, ": ")
    _show_jac_from_to(io, j)
    print(io, ')')
end
