struct BayesianModel{NT<:NamedTuple, PR<:Tuple, SH<:Tuple, TF<:AbstractFloat,
        GJ<:GEJacobian{TF}, TC<:AbstractAllCovCache, TE<:Union{AbstractVector{TF},Nothing},
        GC<:GradientCache, FDA<:NamedTuple, N1, N2, N3}
    shockses::NTuple{N1,Symbol}
    shockparas::NTuple{N2,Symbol}
    modelparas::NTuple{N3,Symbol}
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
        priors::ValidVarInput, data; nTtrim::Integer=20,
        measurement_error::Union{Diagonal,UniformScaling,Nothing}=nothing,
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
    nshpara = length(lookuppara) - nsh
    for k in keys(lookuppara)
        k in gj.tjac.vars && throw(ArgumentError(
            "shock parameter $k coincides with a model parameter"))
    end
    i1 = 0
    priors isa Pair && (priors = (priors,))
    npara = length(priors)
    vpriors = Vector{Distribution}(undef, npara)
    modelparas = Vector{Symbol}(undef, npara-length(lookuppara))
    for (v, p) in priors
        tp = get(lookuppara, v, nothing)
        if tp === nothing
            v in gj.tjac.vars || throw(ArgumentError(
                "$v is not a parameter reachable by the GEJacobian"))
            i1 += 1
            i2 = nsh+nshpara+i1
            vpriors[i2] = p
            modelparas[i1] = v
            lookuppara[v] = (2, i1, i2)
        else
            _, _, i2 = tp
            vpriors[i2] = p
        end
    end
    priors = (vpriors...,)
    modelparas = (modelparas...,)
    paras = (shockses..., shockparas..., modelparas...)
    paravals = NamedTuple{paras}(map(mean, priors))
    observables isa Union{Symbol, Pair} && (observables = (observables,))
    obs, lookupobs = _check_observables(gj, observables)
    Y, Nobs, Tobs = _check_data(data, obs, varvals)
    _demean!(Y, Nobs, Tobs)
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
    return BayesianModel(shockses, shockparas, modelparas, priors, lookuppara,
        obs, lookupobs, Ref(paravals), gj, shocks, Z, G, GZ, SE,
        allcovcache, V, measurement_error, Y, Ycache, nT, Tobs, dl, dlcache, fdkwargs)
end

nshock(bm::BayesianModel) = typeof(bm).parameters[10]
nshockpara(bm::BayesianModel) = typeof(bm).parameters[11]
nmodelpara(bm::BayesianModel) = typeof(bm).parameters[12]

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

function loglikelihood!(bm::BayesianModel, θ)
    nT = bm.nT
    GZ = bm.GZ
    nsh = nshock(bm)
    paravals = _update_paravals!(bm, θ)
    for k in 1:nshock(bm)
        impulse!(view(bm.Z,:,k), bm.shocks[k], paravals)
    end
    #! TO DO: Allow updating G
    for k in 1:nsh
        mul!(view(GZ,:,k:k), view(bm.G,:,1+(k-1)*nT:k*nT), view(bm.Z,:,k:k))
    end
    for i in 1:nsh
        @inbounds bm.SE[i] = paravals[i]
    end
    r = allcov!(bm.allcovcache, _reshape(GZ, nT, size(GZ,1)÷nT, nsh), bm.SE)
    _fill_allcov!(bm.V, r, bm.merror)
    return loglikelihood!(bm.V, bm.Y, bm.Ycache)
end

logposterior!(bm::BayesianModel, θ) = loglikelihood!(bm, θ) + logprior(bm, θ)

function _update_fdcache(ca::GradientCache{TF,TC1,TC2,TC3,fdtype,TF,Val(true)},
        fx) where {TF,TC1,TC2,TC3,fdtype}
    return GradientCache{TF,TC1,TC2,TC3,fdtype,TF,Val(true)}(fx, ca.c1, ca.c2, ca.c3)
end

function logposterior_and_gradient!(bm::BayesianModel, θ)
    l = logposterior!(bm, θ)
    f = Base.Fix1(logposterior!, bm)
    ca = _update_fdcache(bm.dlcache, l)
    finite_difference_gradient!(bm.dl, f, θ, ca; bm.fdkwargs...)
    return l, bm.dl
end

capabilities(::Type{<:BayesianModel}) = LogDensityOrder{1}()
dimension(bm::BayesianModel) = length(bm.priors)
logdensity(bm::BayesianModel, θ) = logposterior!(bm, θ)
function logdensity_and_gradient(bm::BayesianModel, θ)
    l, dl = logposterior_and_gradient!(bm, θ)
    return l, copy(dl)
end
