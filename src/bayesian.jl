struct BayesianModel{NT<:NamedTuple, PR<:Tuple, SH<:Tuple, TF<:AbstractFloat,
        GJ<:GEJacobian{TF}, TC<:AbstractAllCovCache, TE<:Union{AbstractVector{TF},Nothing},
        GC<:GradientCache, HC<:HessianCache, FDA<:NamedTuple, N1, N2, N3}
    shockses::NTuple{N1,Symbol}
    shockparas::NTuple{N2,Symbol}
    strucparas::NTuple{N3,Symbol}
    priors::PR
    lookuppara::Dict{Symbol,NTuple{3,Int}}
    observables::Vector{Pair}
    lookupobs::Dict{Symbol,Int}
    paravals::RefValue{NT}
    gj::GJ
    shocks::SH
    Z::Matrix{TF}
    G::Matrix{TF}
    GZ::Matrix{TF}
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
    vars = gj.tjac.vars
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

function _fillG!(G::Matrix, gj::GEJacobian, observables, nT::Int)
    nTfull = gj.nTfull
    for (j, z) in enumerate(gj.exovars)
        i0 = 0
        for (o, _) in observables
            M = getM!(gj, z, o)
            N = size(M,1) ÷ nTfull
            for _ in 1:N
                copyto!(G, i0+1:i0+nT, 1+(j-1)*nT:j*nT, M, 1:nT, 1:nT)
                i0 += nT
            end
        end
    end
end

function bayesian(gj::GEJacobian{TF}, shocks, observables,
        priors::ValidVarInput, data;
        measurement_error::Union{Diagonal,UniformScaling,Nothing}=nothing,
        nTtrim::Integer=20, demean::Bool=true,
        fdtype=Val(:forward), fdkwargs=NamedTuple(),
        allcovcachekwargs=NamedTuple()) where TF
    shocks isa ShockProcess && (shocks = (shocks,))
    nsh = length(shocks)
    shockses = ntuple(i->shockse(shocks[i]), nsh)
    for sh in shocks
        z = outputs(sh)[1]
        z in gj.exovars || throw(ArgumentError("$z is not an exogenous variable"))
    end
    nexo = length(gj.exovars)
    varvals = gj.tjac.varvals
    nexo == _getvarlength(gj.exovars, varvals) || throw(ArgumentError(
        "not all exogenous variables are scalars"))
    nexo == nsh || throw(ArgumentError(
        "shock process is not specified for every exogenous variable"))
    # The tuple corresponds to parameter type, index within type and index among all
    lookuppara = Dict{Symbol,Tuple{Int,Int,Int}}(v=>(0,i,i) for (i,v) in enumerate(shockses))
    length(lookuppara) == nsh || throw(ArgumentError(
        "names for shock standard errors are not unique"))
    shocks = (shocks...,)
    shockparas = Symbol[]
    i = 0
    for s in shocks
        for v in inputs(s)
            haskey(lookuppara, v) && throw(ArgumentError(
                "name of shock process parameter ($v) is not unique"))
            push!(shockparas, v)
            i += 1
            lookuppara[v] = (1,i,nsh+i)
        end
    end
    shockparas = (shockparas...,)
    nshpara = length(lookuppara)
    for k in keys(lookuppara)
        k in gj.tjac.vars && throw(ArgumentError(
            "name of shock parameter $k coincides with a structural parameter"))
    end
    i1 = 0
    priors isa Pair && (priors = (priors,))
    npara = length(priors)
    vpriors = Vector{Distribution}(undef, npara)
    strucparas = Vector{Symbol}(undef, npara-length(lookuppara))
    for (v, p) in priors
        tp = get(lookuppara, v, nothing)
        if tp === nothing
            v in gj.tjac.vars || throw(ArgumentError(
                "$v is not a parameter reachable by the GEJacobian"))
            i1 += 1
            i2 = nshpara + i1
            vpriors[i2] = p
            strucparas[i1] = v
            lookuppara[v] = (2, i1, i2)
        else
            _, _, i2 = tp
            vpriors[i2] = p
        end
    end
    priors = (vpriors...,)
    strucparas = (strucparas...,)
    paras = (shockses..., shockparas..., strucparas...)
    paravals = NamedTuple{paras}(map(mean, priors))
    observables isa Union{Symbol, Pair} && (observables = (observables,))
    obs, lookupobs = _check_observables(gj, observables)
    Y, Nobs, Tobs = _check_data(data, obs, varvals)
    demean && _demean!(Y, Nobs, Tobs)
    nY = length(Y)
    Ycache = similar(Y)
    nT = gj.tjac.nT - nTtrim
    Z = Matrix{TF}(undef, nT, nsh)
    G = Matrix{TF}(undef, nT*Nobs, length(Z))
    _fillG!(G, gj, obs, nT)
    GZ = Matrix{TF}(undef, size(G, 1), nsh)
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
        obs, lookupobs, Ref(paravals), gj, shocks, Z, G, GZ, SE,
        allcovcache, V, measurement_error, Y, Ycache, nT, Tobs,
        dl, dlcache, d2l, d2lcache, fdkwargs)
end

nshock(bm::BayesianModel) = typeof(bm).parameters[11]
nshockpara(bm::BayesianModel) = typeof(bm).parameters[12] + nshock(bm)
nstrucpara(bm::BayesianModel) = typeof(bm).parameters[13]

const TransformedBayesianModel{T,L} =
    TransformedLogDensity{T,L} where {T<:AbstractTransform, L<:BayesianModel}
const BayesOrTrans = Union{BayesianModel, TransformedBayesianModel}

transform(transformation, bm::BayesianModel) = TransformedLogDensity(transformation, bm)

parent(bm::BayesianModel) = bm
parent(bm::TransformedBayesianModel) = bm.log_density_function

getindex(bm::BayesOrTrans) = parent(bm).paravals[]
getindex(bm::BayesOrTrans, i) = getindex(bm[], i)

function logprior(bm::BayesianModel{NT,PR}, θ) where {NT,PR}
    # The generated part avoids allocations for array θ
    # It is unnecessary if θ is tuple
    if @generated
        ex = :(logpdf(bm.priors[1], θ[1]))
        N = length(PR.parameters)
        if N > 1
            for i in 2:N
                ex = :($ex + logpdf(bm.priors[$i], θ[$i]))
            end
        end
        return ex
    else
        return sum(map(logpdf, bm.priors, θ))
    end
end

logprior(bm::TransformedBayesianModel, θ) =
    logprior(parent(bm), transform(bm.transformation, θ))

function _update_paravals!(bm::BayesianModel{NT}, θ::AbstractVector) where NT
    N = length(NT.parameters[1])
    vals = NamedTuple{NT.parameters[1]}(ntuple(i->θ[i], N))
    bm.paravals[] = vals
    return vals
end

function _update_paravals!(bm::BayesianModel{NT}, θ::Tuple) where NT
    vals = NamedTuple{NT.parameters[1]}(θ)
    bm.paravals[] = vals
    return vals
end

function _update_paravals!(bm::BayesianModel{NT}, θ::NT) where NT
    bm.paravals[] = θ
    return θ
end

function _fill_shocks!(bm::BayesianModel{NT,PR,SH}) where {NT,PR,SH}
    if @generated
        ex = :(impulse!(view(bm.Z,:,1), bm.shocks[1], bm[]))
        N = length(SH.parameters)
        if N > 1
            for i in 2:N
                ex = :($ex; impulse!(view(bm.Z,:,$i), bm.shocks[$i], bm[]))
            end
        end
        ex = :($ex; nothing)
        return ex
    else
        for k in 1:nshock(bm)
            impulse!(view(bm.Z,:,k), bm.shocks[k], bm[])
        end
        return nothing
    end
end

function loglikelihood!(bm::BayesianModel, θ)
    nT = bm.nT
    GZ = bm.GZ
    nsh = nshock(bm)
    paravals = _update_paravals!(bm, θ)
    _fill_shocks!(bm)
    #! TO DO: Allow updating G
    @inbounds for k in 1:nsh
        mul!(view(GZ,:,k), view(bm.G,:,1+(k-1)*nT:k*nT), view(bm.Z,:,k))
    end
    for i in 1:nsh
        @inbounds bm.SE[i] = paravals[i]
    end
    r = allcov!(bm.allcovcache, _reshape(GZ, nT, size(GZ,1)÷nT, nsh), bm.SE)
    _fill_allcov!(bm.V, r, bm.merror)
    return loglikelihood!(bm.V, bm.Y, bm.Ycache)
end

# This method is not used by logposterior!
loglikelihood!(bm::TransformedBayesianModel, θ) =
    transform_logdensity(bm.transformation, Base.Fix1(loglikelihood!, parent(bm)), θ)

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

function logposterior_obj!(bm::BayesOrTrans, θ::AbstractVector, grad::AbstractVector)
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
    f = Base.Fix1(logdensity, bm)
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
    f = Base.Fix1(logdensity, bm)
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

show(io::IO, bm::BayesianModel) = print(io, bm.Tobs, '×', length(bm.observables),
    " BayesianModel(", nshockpara(bm), ", ", nstrucpara(bm), ')')

function show(io::IO, ::MIME"text/plain", bm::BayesianModel)
    nsh = nshockpara(bm)
    nstruc = nstrucpara(bm)
    print(io, bm.Tobs, '×', length(bm.observables),
        " BayesianModel with ", nsh, " shock parameter")
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
    nsh = nshockpara(p)
    nstruc = nstrucpara(p)
    print(io, p.Tobs, '×', length(p.observables),
        " TransformedBayesianModel of dimension ", dimension(bm),
        " from parent model with ", nsh, " shock parameter")
    nsh > 1 && print(io, 's')
    print(io, " and ", nstruc, " structural parameter")
    nstruc > 1 && print(io, 's')
end
