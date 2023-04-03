struct ImpulseUpdate{TF, NT, G<:GMaps{TF}, P<:GEJacobianUpdatePlan}
    gs::G
    paravals::RefValue{NT}
    exovars::Vector{Pair{Symbol,Int}}
    endovars::Vector{Pair{Symbol,Int}}
    npara::Int
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
    npara = _getvarlength(params, paravals)
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
    return ImpulseUpdate(gs, Ref(paravals), exos, endos, npara, nT, p, jacmaps, vals)
end

function _update_paravals!(refvals::RefValue{NT}, θ::AbstractVector) where NT
    if @generated
        names = NT.parameters[1]
        eltypes = NT.parameters[2].parameters
        N = length(names)
        ex = :(())
        excopys = :(vals0 = refvals[])
        for i in 1:N
            if eltypes[i] <: Number
                push!(ex.args, :(θ[$i]))
            else
                excopys = quote
                    $excopys
                    dest = vals0[$i]
                    copyto!(dest, view(θ, $i:$i+length(dest)-1))
                end
                push!(ex.args, :(vals0[$i]))
            end
        end
        ex = :($excopys; vals = NamedTuple{$names}($ex); refvals[] = vals)
        return ex
    else
        names = NT.parameters[1]
        eltypes = NT.parameters[2].parameters
        vals0 = refvals[]
        vs = ()
        for i in 1:length(names)
            if eltypes[i] <: Number
                vs = (vs..., θ[i])
            else
                dest = vals0[i]
                copyto!(dest, view(θ, i:i+length(dest)-1))
                vs = (vs..., dest)
            end
        end
        vals = NamedTuple{names}(vs)
        refvals[] = vals
        return vals
    end
end

function _update_paravals!(refvals::RefValue{NT}, θ::Tuple) where NT
    vals = NamedTuple{NT.parameters[1]}(θ)
    refvals[] = vals
    return vals
end

function _update_paravals!(refvals::RefValue{NT}, θ::NT) where NT
    refvals[] = θ
    return θ
end

function _update_paravals!(refvals::RefValue{NT}, θ::NamedTuple) where NT
    θ1 = NamedTuple{NT.parameters[1]}(θ)
    refvals[] = θ1
    return θ1
end

@inline function (u::ImpulseUpdate)(θ)
    paravals = _update_paravals!(u.paravals, θ)
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
    '×', u.npara, " ImpulseUpdate{", TF, "}(", length(u.exovars), ')')

function show(io::IO, ::MIME"text/plain", u::ImpulseUpdate{TF,NT}) where {TF,NT}
    nexo = length(u.exovars)
    print(io, length(u.vals), '×', u.npara, " ImpulseUpdate{", TF, "} with ")
    println(io, nexo, " exogenous variable", nexo > 1 ? "s:" : ':')
    print(io, "  parameter")
    u.npara > 1 && print(io, 's')
    print(io, ": ")
    join(io, NT.parameters[1], ", ")
end
