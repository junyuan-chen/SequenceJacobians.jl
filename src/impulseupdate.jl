struct ImpulseUpdate{TF, NT, A<:Axis, G<:GMaps{TF}, P<:GEJacobianUpdatePlan}
    gs::G
    paravals::RefValue{NT}
    paraaxis::A
    exovars::Vector{Pair{Symbol,Int}}
    endovars::Vector{Pair{Symbol,Int}}
    nT::Int
    plan::P
    jacmaps::Vector{AbstractJacobianMap{TF}}
    vals::Array{TF,3}
end

@inline function _fillG1!(G::Array{T,3}, jacmaps::Vector, exos::Vector{Pair{Symbol,Int}},
        endos::Vector{Pair{Symbol,Int}}) where T
    nT = size(G, 1)
    k, j0 = 1, 0
    for (_, wz) in exos
        Gz = selectdim(G, 3, j0+1:j0+wz)
        i0 = 0
        for (_, wu) in endos
            mul!(_reshape(selectdim(Gz, 2, i0+1:i0+wu), nT*wu, wz), jacmaps[k], true, mb=wu)
            i0 += wu
            k += 1
        end
        j0 += wz
    end
end

function ImpulseUpdate(gs::GMaps{TF}, params, exovars, endovars, nT::Int;
        dZvars=nothing) where TF
    gj = gs.gj
    params isa Symbol && (params = (params,))
    exovars isa Symbol && (exovars = (exovars,))
    endovars isa Symbol && (endovars = (endovars,))
    params = ntuple(i->params[i], length(params))
    nT > gj.nTfull && throw(ArgumentError("nT cannot be greater than $(gj.nTfull)"))
    paravals = NamedTuple{params}(gj.tjac.varvals[])
    paraaxis = getfield(ComponentVector(paravals), :axes)[1]
    p = plan(gj, params; dZvars=dZvars)
    jacmaps = Vector{AbstractJacobianMap{TF}}(undef, length(exovars)*length(endovars))
    varvals = gj.tjac.varvals[]
    wexo = 0
    wendo = 0
    exos = Vector{Pair{Symbol,Int}}(undef, length(exovars))
    endos = Vector{Pair{Symbol,Int}}(undef, length(endovars))
    k = 1
    for (j, exo) in enumerate(exovars)
        wj = gj.nZcol[findfirst(==(exo), gj.exovars)]
        wexo += wj
        exos[j] = exo=>wj
        dmaps = gs[exo]
        for (i, endo) in enumerate(endovars)
            if j == 1
                wi = length(varvals[endo])
                wendo += wi
                endos[i] = endo=>wi
            end
            jacmaps[k] = dmaps[endo]
            k += 1
        end
    end
    vals = Array{TF,3}(undef, nT, wendo, wexo)
    return ImpulseUpdate(gs, Ref(paravals), paraaxis, exos, endos, nT, p, jacmaps, vals)
end

function _unsafe_update_paravals!(refvals::RefValue{NamedTuple{names, T}},
        θ::ComponentVector, axis=nothing) where {names, T}
    if @generated
        eltypes = T.parameters
        N = length(names)
        ex = :(())
        excopys = :(vals0 = refvals[])
        for i in 1:N
            n = QuoteNode(names[i])
            if eltypes[i] <: Number
                push!(ex.args, :(getproperty(θ, $n)))
            else
                excopys = :($excopys; copyto!(vals0[$n], getproperty(θ, $n)))
                push!(ex.args, :(vals0[$n]))
            end
        end
        ex = :($excopys; vals = NamedTuple{$names}($ex); refvals[] = vals)
        return ex
    else
        eltypes = T.parameters
        vals0 = refvals[]
        vs = ()
        for i in 1:length(names)
            if eltypes[i] <: Number
                vs = (vs..., θ[names[i]])
            else
                dest = vals0[i]
                copyto!(dest, θ[names[i]])
                vs = (vs..., dest)
            end
        end
        vals = NamedTuple{names}(vs)
        refvals[] = vals
        return vals
    end
end

function _unsafe_update_paravals!(refvals::RefValue{NamedTuple{names, T}},
        θ::Tuple, axis=nothing) where {names, T}
    if @generated
        eltypes = T.parameters
        N = length(names)
        ex = :(())
        excopys = :(vals0 = refvals[])
        for i in 1:N
            if eltypes[i] <: Number
                push!(ex.args, :(θ[$i]))
            else
                excopys = :($excopys; copyto!(vals0[$i], θ[$i]))
                push!(ex.args, :(vals0[$i]))
            end
        end
        ex = :($excopys; vals = NamedTuple{$names}($ex); refvals[] = vals)
        return ex
    else
        eltypes = T.parameters
        vals0 = refvals[]
        vs = ()
        for i in 1:length(names)
            if eltypes[i] <: Number
                vs = (vs..., θ[i])
            else
                dest = vals0[i]
                copyto!(dest, θ[i])
                vs = (vs..., dest)
            end
        end
        vals = NamedTuple{names}(vs)
        refvals[] = vals
        return vals
    end
end

@inline _update_paravals!(refvals::RefValue{NT}, θ::AbstractVector, axis::Axis) where NT =
    _unsafe_update_paravals!(refvals, ComponentArray(θ, (axis,)))

@inline _update_paravals!(refvals::RefValue{NT}, θ::NT, axis=nothing) where NT =
    _unsafe_update_paravals!(refvals, (θ...,))

@inline _update_paravals!(refvals::RefValue{NT}, θ::NamedTuple, axis=nothing) where NT =
    _unsafe_update_paravals!(refvals, (NamedTuple{NT.parameters[1]}(θ)...,))

@inline function _update_paravals!(refvals::RefValue{NT}, θ::Tuple, axis=nothing) where NT
    length(NT.parameters[1]) == length(θ) ||
        throw(DimensionMismatch("length of refvals must match length of θ"))
    return _unsafe_update_paravals!(refvals, θ)
end

@inline function (u::ImpulseUpdate)(θ)
    paravals = _update_paravals!(u.paravals, θ, u.paraaxis)
    tjac = u.gs.gj.tjac
    tjac.varvals[] = varvals = merge(tjac.varvals[], paravals)
    u.plan(varvals)
    u.gs()
    _fillG1!(u.vals, u.jacmaps, u.exovars, u.endovars)
    return u
end

@inline getindex(u::ImpulseUpdate) = u.paravals[]
@inline getindex(u::ImpulseUpdate, i) = getindex(u[], i)

show(io::IO, u::ImpulseUpdate{TF}) where TF = print(io, length(u.vals),
    '×', lastindex(u.paraaxis), " ImpulseUpdate{", TF, "}(", length(u.exovars), ')')

function show(io::IO, ::MIME"text/plain", u::ImpulseUpdate{TF,NT}) where {TF,NT}
    nexo = length(u.exovars)
    print(io, length(u.vals), '×', lastindex(u.paraaxis), " ImpulseUpdate{", TF, "} with ")
    println(io, nexo, " exogenous variable", nexo > 1 ? "s:" : ':')
    print(io, "  parameter")
    lastindex(u.paraaxis) > 1 && print(io, 's')
    print(io, ": ")
    join(io, NT.parameters[1], ", ")
end

struct PEImpulseUpdate{TF, NT, A<:Axis, J<:PECombinedBlockJacobian}
    j::J
    paravals::RefValue{NT}
    paraaxis::A
    exovars::Vector{Pair{Symbol,Int}}
    endovars::Vector{Pair{Symbol,Int}}
    nT::Int
    jacmaps::Vector{AbstractJacobianMap{TF}}
    vals::Array{TF,3}
end

function PEImpulseUpdate(j::PECombinedBlockJacobian{BLK,TF}, params, exovars, endovars,
        nT::Int) where {BLK,TF}
    params isa Symbol && (params = (params,))
    exovars isa Symbol && (exovars = (exovars,))
    endovars isa Symbol && (endovars = (endovars,))
    params = ntuple(i->params[i], length(params))
    tjac = j.tjac
    nT > tjac.nT && throw(ArgumentError("nT cannot be greater than $(tjac.nT)"))
    varvals = tjac.varvals[]
    paravals = NamedTuple{params}(varvals)
    paraaxis = getfield(ComponentVector(paravals), :axes)[1]
    jacmaps = Vector{AbstractJacobianMap{TF}}(undef, length(exovars)*length(endovars))
    wexo = 0
    wendo = 0
    exos = Vector{Pair{Symbol,Int}}(undef, length(exovars))
    endos = Vector{Pair{Symbol,Int}}(undef, length(endovars))
    k = 1
    for (j, exo) in enumerate(exovars)
        wj = tjac.ncol[findfirst(==(exo), tjac.srcs)]
        wexo += wj
        exos[j] = exo=>wj
        dmaps = tjac.totals[exo]
        for (i, endo) in enumerate(endovars)
            if j == 1
                wi = length(varvals[endo])
                wendo += wi
                endos[i] = endo=>wi
            end
            jacmaps[k] = dmaps[endo]
            k += 1
        end
    end
    vals = Array{TF,3}(undef, nT, wendo, wexo)
    return PEImpulseUpdate(j, Ref(paravals), paraaxis, exos, endos, nT, jacmaps, vals)
end

@inline function (u::PEImpulseUpdate)(θ)
    paravals = _update_paravals!(u.paravals, θ, u.paraaxis)
    tjac = u.j.tjac
    tjac.varvals[] = varvals = merge(tjac.varvals[], paravals)
    u.j(varvals)
    _fillG1!(u.vals, u.jacmaps, u.exovars, u.endovars)
    return u
end

@inline getindex(u::PEImpulseUpdate) = u.paravals[]
@inline getindex(u::PEImpulseUpdate, i) = getindex(u[], i)

show(io::IO, u::PEImpulseUpdate{TF}) where TF = print(io, length(u.vals),
    '×', lastindex(u.paraaxis), " PEImpulseUpdate{", TF, "}(", length(u.exovars), ')')

function show(io::IO, ::MIME"text/plain", u::PEImpulseUpdate{TF,NT}) where {TF,NT}
    nexo = length(u.exovars)
    print(io, length(u.vals), '×', lastindex(u.paraaxis), " PEImpulseUpdate{", TF, "} with ")
    println(io, nexo, " exogenous variable", nexo > 1 ? "s:" : ':')
    print(io, "  parameter")
    lastindex(u.paraaxis) > 1 && print(io, 's')
    print(io, ": ")
    join(io, NT.parameters[1], ", ")
end
