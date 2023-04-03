abstract type AbstractEstimator{TF<:AbstractFloat, NT<:NamedTuple} end

struct BayesianModel{TF, NT, PR<:Tuple, SH<:Tuple, N1, N2, N3,
        U<:Union{ImpulseUpdate, Nothing},
        GJ<:GEJacobian{TF}, TC<:AbstractAllCovCache, TE<:Union{AbstractVector{TF},Nothing},
        GC<:GradientCache, HC<:HessianCache, FDA<:NamedTuple} <: AbstractEstimator{TF, NT}
    shockses::NTuple{N1,Symbol}
    shockparas::NTuple{N2,Symbol}
    strucparas::NTuple{N3,Symbol}
    priors::PR
    lookuppara::Dict{Symbol,NTuple{4,Int}}
    exovars::Vector{Symbol}
    observables::Vector{Pair}
    lookupobs::Dict{Symbol,Int}
    paravals::RefValue{NT}
    impulseupdate::U
    gs::GMaps{TF,GJ}
    shocks::SH
    Z::Vector{Vector{TF}}
    G::Array{TF,3}
    GZ::Array{TF,3}
    SE::Vector{TF}
    allcovcache::TC
    V::Matrix{TF}
    merror::TE
    Y::Vector{TF}
    Ycache::Vector{TF}
    nT::Int
    Tobs::Int
    dl::Vector{TF}
    dlcache::GC
    d2l::Matrix{TF}
    d2lcache::HC
    fdkwargs::FDA
end

function _check_observables(gj::GEJacobian, observables)
    vars = gj.tjac.pool
    exos = gj.exovars
    obs = Vector{Pair}(undef, length(observables))
    lookup = Dict{Symbol,Int}()
    for (i, p) in enumerate(observables)
        if p isa Pair
            obs[i] = p
            lookup[p[1]] = i
        elseif p isa Symbol
            obs[i] = p=>p
            lookup[p] = i
        else
            throw(ArgumentError("$p is not accepted for specifying observables"))
        end
    end
    length(lookup) == length(obs) || throw(ArgumentError("observable names are not unique"))
    for (v1, _) in obs
        v1 in vars || throw(ArgumentError(
            "$v1 is not a variable reachable by the GEJacobian"))
        v1 in exos && throw(ArgumentError("$v1 is an exogenous variable"))
    end
    return obs, lookup
end

function _check_data(data::AbstractVector, observables, vals)
    N = _getvarlength((p[1] for p in observables), vals)
    T = length(data) ÷ N
    T * N == length(data) || throw(ArgumentError(
        "length of data ($(length(data))) is not a multiple of number of observables ($N)"))
    return data, N, T
end

# Assume matrix columns are in the correct order
function _check_data(data::AbstractMatrix, observables, vals)
    T, N = size(data)
    N1 = _getvarlength((p[1] for p in observables), vals)
    N == N1 || throw(ArgumentError(
        "number of data columns ($N) does not match number of observables ($N1)"))
    return vec(data), N, T
end

function _check_data(data, observables, vals)
    Tables.istable(data) || throw(ArgumentError("data of type $(typeof(data)) is not accepted; require a vector, matrix or `Tables.jl`-compatible table"))
    T = Tables.rowcount(data)
    N = _getvarlength((p[1] for p in observables), vals)
    Y = Vector{Float64}(undef, N*T)
    i0 = 0
    for i in eachindex(observables)
        n = observables[i][2]
        if n isa Union{Symbol,Int}
            col = Tables.getcolumn(data, n)
            length(col) == T || error("length of data column $n is not $T")
            copyto!(Y, 1+i0*T, col)
            i0 += 1
        else
            for v in n
                col = Tables.getcolumn(data, v)
                length(col) == T || error("length of data column $v is not $T")
                copyto!(Y, 1+i0*T, col)
                i0 += 1
            end
        end
    end
    return Y, N, T
end

function _demean!(Y, N::Int, T::Int)
    M = _reshape(Y, T, N)
    for y in eachcol(M)
        y .-= mean(y)
    end
end

function _fillGfull!(G::AbstractArray{T,3}, gs::GMaps, exos::Vector{Pair{Symbol,Int}},
        endos::Vector{Pair{Symbol,Int}}) where T
    nT = size(G, 2)
    j0 = 0
    for (z, wz) in exos
        Gz = selectdim(G, 3, j0+1:j0+wz)
        i0 = 0
        for (u, wu) in endos
            mul!(_reshape(selectdim(Gz, 1, i0+1:i0+wu*nT), nT*wu, nT*wz), gs[z,u], true;
                mb=wu, nb=wz)
            i0 += wu*nT
        end
        j0 += wz
    end
end

function _setparaname!(d::Dict, v::Vector, n::Symbol, itype::Int, iall::Int, s::Int)
    haskey(d, n) && throw(ArgumentError("shock parameter name $n is not unique"))
    push!(v, n)
    iall += 1
    d[n] = (itype, length(v), iall, s)
    return iall
end

_mean(d::Distribution) = mean(d)
_mean(ds::StructArray{<:Distribution}) = map(mean, ds)

function bayesian(gs::GMaps{TF}, shocks, observables,
        priors::ValidVarInput, data;
        measurement_error::Union{Diagonal,UniformScaling,Nothing}=nothing,
        nTtrim::Integer=20, demean::Bool=true,
        fdtype=Val(:forward), fdkwargs=NamedTuple(),
        allcovcachekwargs=NamedTuple()) where TF
    gj = gs.gj
    varvals = gj.tjac.varvals[]
    shocks isa ShockProcess && (shocks = (shocks,))
    nsh = length(shocks)
    exovars = Vector{Symbol}(undef, nsh)
    # The tuple corresponds to parameter type, index within type, index among all
    # and index for the associated shock (if positive)
    lookuppara = Dict{Symbol,Tuple{Int,Int,Int,Int}}()
    shockses = Symbol[]
    shockparas = Symbol[]
    npara = 0
    for s in 1:nsh
        sh = shocks[s]
        z = outputs(sh)[1]
        z in gj.exovars || throw(ArgumentError("$z is not an exogenous variable"))
        exovars[s] = z
        npara = _setparaname!(lookuppara, shockses, shockse(sh), 0, npara, s)
    end
    # Loop the other parameters separately to keep indices for shockses in the front
    for s in 1:nsh
        for n in inputs(shocks[s])
            npara = _setparaname!(lookuppara, shockparas, n, 1, npara, s)
        end
    end
    for k in keys(lookuppara)
        haskey(gj.tjac.invpool, k) && throw(ArgumentError(
            "name of shock parameter $k coincides with a structural parameter"))
    end
    shocks = (shocks...,)
    shockses = (shockses...,)
    shockparas = (shockparas...,)

    priors isa Pair && (priors = (priors,))
    vpriors = Vector{Any}(undef, npara)
    strucparas = Symbol[]
    for (v, p) in priors
        # Assume there is no duplicate of parameter name in priors
        tp = get(lookuppara, v, nothing)
        n = tp === nothing ? v : exovars[tp[4]]
        if p isa Distribution
            length(varvals[n]) == 1 || throw(ArgumentError(
                "$v should match only one prior distribution"))
        elseif p isa StructVector{<:Distribution}
            length(p) == length(varvals[n]) || throw(ArgumentError(
                "$v should match $(length(varvals[n])) prior distributions"))
        else
            throw(ArgumentError("element of type $(typeof(p)) is not accepted for priors"))
        end
        if tp === nothing
            push!(vpriors, p)
            push!(strucparas, v)
            lookuppara[v] = (2, length(strucparas), length(vpriors), 0)
        else
            _, _, i2, _ = tp
            vpriors[i2] = p
        end
    end
    dZs = gj.tjac.dZs
    if isempty(strucparas)
        dZs === nothing || throw(ArgumentError(
            "TotalJacobian with dZs is not allowed when estimation only involves shock parameters"))
    else
        dZs === nothing && throw(ArgumentError(
            "TotalJacobian requires dZs when estimation involves structural parameters"))
        for exo in exovars
            dZ = get(dZs, exo, nothing)
            dZ === nothing && throw(ArgumentError(
                "dZs is not specified for $exo in TotalJacobian"))
            size(dZ, 2) === 1 || throw(ArgumentError(
                "dZ for $exo contains multiple columns"))
        end
    end
    priors = (vpriors...,)
    strucparas = (strucparas...,)
    paras = (shockses..., shockparas..., strucparas...)
    paravals = NamedTuple{paras}(map(_mean, priors))

    observables isa Union{Symbol, Pair} && (observables = (observables,))
    obs, lookupobs = _check_observables(gj, observables)
    Y, Nobs, Tobs = _check_data(data, obs, varvals)
    demean && _demean!(Y, Nobs, Tobs)
    nY = length(Y)
    Ycache = similar(Y)
    nT = gj.tjac.nT - nTtrim
    if isempty(strucparas)
        u = nothing
        Z = Vector{Vector{TF}}(undef, nsh)
        wexo = 0
        wendo = 0
        exos = Vector{Pair{Symbol,Int}}(undef, length(exovars))
        endos = Vector{Pair{Symbol,Int}}(undef, length(observables))
        for (j, exo) in enumerate(exovars)
            wj = length(varvals[exo])
            wexo += wj
            exos[j] = exo=>wj
            Z[j] = Vector{TF}(undef, wj*nT)
        end
        for (i, (endo, _)) in enumerate(observables)
            wi = length(varvals[endo])
            wendo += wi
            endos[i] = endo=>wi
        end
        G = Array{TF,3}(undef, wendo*nT, nT, wexo)
        _fillGfull!(G, gs, exos, endos)
        GZ = Array{TF,3}(undef, nT, wendo, wexo)
    else
        u = ImpulseUpdate(gs, strucparas, exovars, map(Fix2(getindex, 1), observables), nT;
            dZvars=exovars)
        Z = [reshape(dZs[z], length(dZs[z])) for z in exovars]
        G = Array{TF,3}(undef, (0,0,0))
        GZ = u.vals
    end
    SE = Vector{TF}(undef, nsh)
    allcovcache = FFTWAllCovCache(nT, Nobs, nsh, TF; allcovcachekwargs...)
    V = Matrix{TF}(undef, nY, nY)
    dl = Vector{TF}(undef, npara)
    dlcache = GradientCache{TF,Nothing,Nothing,Vector{TF},fdtype,TF,Val(true)}(
        zero(TF), nothing, nothing, similar(dl))
    d2l = Matrix{TF}(undef, npara, npara)
    d2lcache = HessianCache{Vector{TF},Val(:hcentral),Val(true)}(
        similar(dl), similar(dl), similar(dl), similar(dl))
    return BayesianModel(shockses, shockparas, strucparas, priors, lookuppara,
        exovars, obs, lookupobs, Ref(paravals), u, gs, shocks, Z, G, GZ, SE,
        allcovcache, V, measurement_error, Y, Ycache, nT, Tobs,
        dl, dlcache, d2l, d2lcache, fdkwargs)
end

nshock(bm::BayesianModel) = typeof(bm).parameters[5]
nshockpara(bm::BayesianModel) = typeof(bm).parameters[6] + nshock(bm)
nstrucpara(bm::BayesianModel) = typeof(bm).parameters[7]

const TransformedBayesianModel{T,L} =
    TransformedLogDensity{T,L} where {T<:AbstractTransform, L<:BayesianModel}
const BayesOrTrans = Union{BayesianModel, TransformedBayesianModel}

transform(transformation, bm::BayesianModel) = TransformedLogDensity(transformation, bm)

parent(bm::BayesianModel) = bm
parent(bm::TransformedBayesianModel) = bm.log_density_function

@inline getindex(bm::BayesOrTrans) = parent(bm).paravals[]
@inline getindex(bm::BayesOrTrans, i) = getindex(bm[], i)

_logpdf(d::Distribution, θ::Number) = logpdf(d, θ)
function _logpdf(ds::StructArray{<:Distribution}, θs::Number)
    s = zero(eltype(θs))
    @inbounds for i in eachindex(ds)
        s += logpdf(ds[i], θs[i])
    end
    return s
end

function logprior(bm::BayesianModel{TF,NT,PR}, θ::AbstractArray) where {TF,NT,PR}
    # The generated part avoids allocations for array θ
    if @generated
        ptypes = PR.parameters
        if ptypes[1] <: Distribution
            ex = :(lp = _logpdf(bm.priors[1], θ[1]); i0 = 2)
        else
            ex = quote
                p1 = bm.priors[1]
                lp = _logpdf(p1, view(θ, 1:length(p1)))
                i0 = length(p1) + 1
            end
        end
        N = length(ptypes)
        if N > 1
            for i in 2:N
                if ptypes[i] <: Distribution
                    ex = :($ex; lp += _logpdf(bm.priors[$i], θ[i0]); i0 += 1)
                else
                    pp = Symbol(:p, i)
                    ex = quote
                        $ex
                        $pp = bm.priors[$i]
                        lp += _logpdf($pp, view(θ, i0:i0+length($pp)-1))
                        i0 += length($pp)
                    end
                end
            end
        end
        return :($ex; lp)
    else
        ptypes = PR.parameters
        if ptypes[1] <: Distribution
            lp = _logpdf(bm.priors[1], θ[1])
            i0 = 2
        else
            p1 = bm.priors[1]; lp = _logpdf(p1, view(θ, 1:length(p1)))
            i0 = length(p1) + 1
        end
        N = length(ptypes)
        if N > 1
            for i in 2:N
                if ptypes[i] <: Distribution
                    lp += _logpdf(bm.priors[i], θ[i0])
                    i0 += 1
                else
                    pp = bm.priors[i]
                    lp += _logpdf(pp, view(θ, i:i+length(pp)-1))
                    i0 += length(pp)
                end
            end
        end
        return lp
    end
end

# Fallback method is intended to handle NamedTuple and Tuple θ
logprior(bm::BayesianModel{TF,NT,PR}, θ) where {TF,NT,PR} =
    sum(map(_logpdf, bm.priors, θ))

logprior(bm::TransformedBayesianModel, θ) =
    logprior(parent(bm), transform(bm.transformation, θ))

function _fill_shocks!(bm::BayesianModel{TF,NT,PR,SH}) where {TF,NT,PR,SH}
    if @generated
        ex = :(impulse!(bm.Z[1], bm.shocks[1], bm[]))
        N = length(SH.parameters)
        for i in 2:N
            ex = :($ex; impulse!(bm.Z[$i], bm.shocks[$i], bm[]))
        end
        ex = :($ex; nothing)
        return ex
    else
        for k in 1:nshock(bm)
            impulse!(bm.Z[k], bm.shocks[k], bm[])
        end
        return nothing
    end
end

function loglikelihood!(bm::BayesianModel{TF,NT,PR,SH,N1,N2,0}, θ) where {TF,NT,PR,SH,N1,N2}
    GZ = bm.GZ
    nsh = nshock(bm)
    paravals = _update_paravals!(bm.paravals, θ)
    _fill_shocks!(bm)
    @inbounds for k in 1:nsh
        GZk = selectdim(GZ,3,k)
        mul!(_reshape(GZk, length(GZk)), selectdim(bm.G,3,k), bm.Z[k])
    end
    for i in 1:nsh
        @inbounds bm.SE[i] = paravals[i]
    end
    r = allcov!(bm.allcovcache, GZ, bm.SE)
    _fill_allcov!(bm.V, r, bm.merror)
    return loglikelihood!(bm.V, bm.Y, bm.Ycache)
end

function loglikelihood!(bm::BayesianModel, θ)
    nsh = nshock(bm)
    paravals = _update_paravals!(bm.paravals, θ)
    _fill_shocks!(bm)
    # Update impulse responses
    bm.impulseupdate(paravals) # Structural parameters in tjac are updated
    for i in 1:nsh
        @inbounds bm.SE[i] = paravals[i]
    end
    r = allcov!(bm.allcovcache, bm.GZ, bm.SE)
    _fill_allcov!(bm.V, r, bm.merror)
    return loglikelihood!(bm.V, bm.Y, bm.Ycache)
end

# This method is not used by logposterior!
loglikelihood!(bm::TransformedBayesianModel, θ) =
    transform_logdensity(bm.transformation, Fix1(loglikelihood!, parent(bm)), θ)

logposterior!(bm::BayesianModel, θ) = loglikelihood!(bm, θ) + logprior(bm, θ)
# Does not add the log Jacobian determinant
logposterior!(bm::TransformedBayesianModel, θ) =
    logposterior!(parent(bm), transform(bm.transformation, θ))

# Needed for TransformedLogDensity
(bm::BayesOrTrans)(θ) = logposterior!(bm, θ)

function _update_fdcache(ca::GradientCache{TF,TC1,TC2,TC3,fdtype,TF,Val(true)},
        fx) where {TF,TC1,TC2,TC3,fdtype}
    return GradientCache{TF,TC1,TC2,TC3,fdtype,TF,Val(true)}(fx, ca.c1, ca.c2, ca.c3)
end

_resize_fdcache!(ca::GradientCache{<:Any,Nothing,Nothing,<:AbstractVector}, N::Int) =
    resize!(ca.c3, N)

function logposterior_and_gradient!(bm::BayesOrTrans, θ, grad::AbstractVector)
    l = logposterior!(bm, θ)
    p = parent(bm)
    ca = p.dlcache
    # Transformation could change the dimension
    dimension(bm) <= length(ca.c3) || _resize_fdcache!(ca, dimension(bm))
    ca = _update_fdcache(ca, l)
    finite_difference_gradient!(grad, bm, θ, ca; p.fdkwargs...)
    return l, grad
end

function logposterior_and_gradient!(bm::BayesOrTrans, θ)
    grad = parent(bm).dl
    dimension(bm) == length(grad) || resize!(grad, dimension(bm))
    return logposterior_and_gradient!(bm, θ, grad)
end

function (bm::BayesOrTrans)(θ, grad::AbstractVector)
    if length(grad) > 0
        l, grad = logposterior_and_gradient!(bm, θ, grad)
        return l
    else
        return logposterior!(bm, θ)
    end
end

capabilities(::Type{<:BayesOrTrans}) = LogDensityOrder{1}()
dimension(bm::BayesianModel) = length(bm.priors)
logdensity(bm::BayesianModel, θ) = logposterior!(bm, θ)

# Automatic differentiation is not applicable and hence use FiniteDiff
function logdensity_and_gradient(bm::BayesianModel, θ)
    l, dl = logposterior_and_gradient!(bm, θ)
    return l, copy(dl)
end

function logdensity_and_gradient(bm::TransformedBayesianModel, θ)
    # logdensity adds the log Jacobian determinant for variable transformation
    l = logdensity(bm, θ)
    p = parent(bm)
    grad = parent(bm).dl
    dimension(bm) == length(grad) || resize!(grad, dimension(bm))
    f = Fix1(logdensity, bm)
    ca = p.dlcache
    dimension(bm) <= length(ca.c3) || _resize_fdcache!(ca, dimension(bm))
    ca = _update_fdcache(ca, l)
    finite_difference_gradient!(grad, f, θ, ca; p.fdkwargs...)
    return l, copy(grad)
end

_resize_fdcache!(ca::HessianCache, N) =
    (resize!(ca.xpp, N); resize!(ca.xpm, N); resize!(ca.xmp, N); resize!(ca.xmm, N))

_update_fdcache!(ca::HessianCache, x) =
    (copyto!(ca.xpp, x); copyto!(ca.xpm, x); copyto!(ca.xmp, x); copyto!(ca.xmm, x))

function logdensity_hessian!(bm::BayesOrTrans, θ, h::AbstractMatrix)
    p = parent(bm)
    ca = p.d2lcache
    # Transformation could change the dimension
    dimension(bm) <= length(ca.xpp) || _resize_fdcache!(ca, dimension(bm))
    _update_fdcache!(ca, θ)
    # Share fdkwargs with gradient for simplicity but their keywords are not identical
    f = Fix1(logdensity, bm)
    finite_difference_hessian!(h, f, θ, ca; p.fdkwargs...)
    return h
end

function logdensity_hessian!(bm::BayesOrTrans, θ)
    h = parent(bm).d2l
    dimension(bm) == size(h,1) || error(DimensionMismatch("cannot use bm.d2l for output"))
    return logdensity_hessian!(bm, θ, h)
end

function vcov!(out::AbstractMatrix, bm::BayesOrTrans, θ)
    p = parent(bm)
    N = dimension(bm)
    size(out) == (N, N) || throw(DimensionMismatch("expect the size of out to be ($N, $N)"))
    logdensity_hessian!(bm, θ, p.d2l)
    TF = eltype(out)
    fill!(out, zero(TF))
    out[diagind(out)] .-= one(TF)
    # One allocation from lu! cannot be avoided
    return ldiv!(lu!(p.d2l), out)
end

vcov(bm::BayesOrTrans, θ) = vcov!(similar(parent(bm).d2l), bm, θ)

function stderror(bm::BayesOrTrans, θ)
    Σ = vcov(bm, θ)
    return map(i->@inbounds(sqrt(Σ[i])), diagind(Σ))
end

show(io::IO, bm::BayesianModel{TF}) where TF =
    print(io, bm.Tobs, '×', length(bm.observables), " BayesianModel{", TF, "}(",
        nshockpara(bm), ", ", nstrucpara(bm), ')')

function show(io::IO, ::MIME"text/plain", bm::BayesianModel{TF}) where TF
    nsh = nshockpara(bm)
    nstruc = nstrucpara(bm)
    print(io, bm.Tobs, '×', length(bm.observables),
        " BayesianModel{", TF, "} with ", nsh, " shock parameter")
    nsh > 1 && print(io, 's')
    print(io, " and ", nstruc, " structural parameter")
    nstruc > 1 && print(io, 's')
    println(io, ":")
    print(io, "  shock parameter")
    nsh > 1 && print(io, 's')
    print(io, ": ")
    join(io, (bm.shockses..., bm.shockparas...), ", ")
    if nstruc > 0
        print(io, "\n  structural parameter")
        nstruc > 1 && print(io, 's')
        print(io, ": ")
        join(io, bm.strucparas, ", ")
    end
end

function show(io::IO, bm::TransformedBayesianModel)
    p = parent(bm)
    print(io, p.Tobs, '×', length(p.observables),
        " TransformedBayesianModel(", dimension(bm), ')')
end

function show(io::IO, ::MIME"text/plain", bm::TransformedBayesianModel)
    p = parent(bm)
    TF = typeof(p).parameters[1]
    nsh = nshockpara(p)
    nstruc = nstrucpara(p)
    print(io, p.Tobs, '×', length(p.observables),
        " TransformedBayesianModel of dimension ", dimension(bm),
        " from BayesianModel{", TF, "} with ", nsh, " shock parameter")
    nsh > 1 && print(io, 's')
    print(io, " and ", nstruc, " structural parameter")
    nstruc > 1 ? println(io, "s:") : println(io, ':')
    trans = replace(sprint(show, MIME("text/plain"), bm.transformation), '\n'=>"\n  ")
    print(io, "  ", trans)
end
