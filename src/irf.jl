const JacType{TF} = Union{AbstractMatrix{TF}, Shift{TF}} where {TF<:AbstractFloat}

struct Jacobians{TF<:AbstractFloat}
    parent::SequenceSpaceModel
    blks::Vector{AbstractBlock}
    vars::Set{Symbol}
    varvals::Dict{Symbol,ValType{TF}}
    T::Int
    unknowns::Set{Symbol}
    targets::Vector{Symbol}
    jacs::IdDict{Symbol,Vector{Pair{Symbol,JacType{TF}}}}
    maps::IdDict{Symbol,IdDict{Symbol,LinearMap}}
end

_tomap(S::Shift, T::Int) = ShiftMap(S, T)
_tomap(M::AbstractMatrix, ::Int) = LinearMap(M)

function Jacobians(m::SequenceSpaceModel, unknowns, targets,
        varvals::Dict{Symbol,ValType{TF}}, T::Int) where TF
    unknowns isa Symbol && (unknowns = (unknowns,))
    targets isa Symbol && (targets = (targets,))
    isempty(unknowns) && throw(ArgumentError("unknowns cannot be empty"))
    isempty(targets) && throw(ArgumentError("targets cannot be empty"))
    unknowns = Set{Symbol}(unknowns)
    targets = collect(Symbol, targets)
    vars = copy(unknowns)
    blks = AbstractBlock[]
    jacs = IdDict{Symbol,Vector{Pair{Symbol,JacType{TF}}}}()
    Dmap = IdDict{Symbol,LinearMap}
    maps = IdDict{Symbol,Dmap}(u=>Dmap() for u in vars)
    for v in m.order
        isblock(m, v) || continue
        blk = m.pool[v]
        blk isa Symbol && continue
        pushed = false
        for (i, vi) in enumerate(inputs(blk))
            if vi in vars
                J = jacobian(blk, i, varvals)
                for (r, vo) in enumerate(outputs(blk))
                    sh = shift(invars(blk)[i])
                    js = get!(valtype(jacs), jacs, vo)
                    # ! to do: consider vector inputs/outputs and block types
                    j = Shift(sh, J[r,1])
                    push!(js, vi=>j)
                    iszero(j) && continue
                    push!(vars, vo)
                    j = _tomap(j, T)
                    # Handle the case when vi is a source of the DAG
                    unknown = get(maps, vi, nothing)
                    if unknown === nothing
                        for d in values(maps)
                            maplast = get(d, vi, nothing)
                            if maplast !== nothing
                                jcomp = j * maplast
                                # vo may exist when temporal terms are involved
                                map = get(d, vo, nothing)
                                d[vo] = map === nothing ? jcomp : map+jcomp
                            end
                        end
                    else
                        map = get(unknown, vo, nothing)
                        unknown[vo] = map === nothing ? j : map+j
                    end
                end
                if !pushed
                    push!(blks, blk)
                    pushed = true
                end
            end
        end
    end
    return Jacobians(m, blks, vars, varvals, T, unknowns, targets, jacs, maps)
end

struct ImpulseResponseMaps{TF<:AbstractFloat}
    jacs::Jacobians{TF}
    exovars::Vector{Symbol}
    endovars::Vector{Symbol}
    Gs::IdDict{Symbol,IdDict{Symbol,Union{Matrix{TF},LinearMap{TF}}}}
end

function ImpulseResponseMaps(jacs::Jacobians{TF}, exovars, endovars) where TF
    exovars isa Symbol && (exovars = (exovars,))
    endovars isa Symbol && (endovars = (endovars,))
    isempty(exovars) && throw(ArgumentError("exovars cannot be empty"))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    exovars = collect(Symbol, exovars)
    endovars = collect(Symbol, endovars)
    for vars in (exovars, endovars)
        for v in vars
            v in jacs.unknowns || throw(ArgumentError("$v is not an unknown variable"))
        end
    end
    T = jacs.T
    Nexo = length(exovars)
    Nendo = length(endovars)
    Ntar = length(jacs.targets)
    H_U = Matrix(hvcat(((Nendo for _ in 1:Ntar)...,),
        (jacs.maps[v][t] for t in jacs.targets for v in endovars)...))
    H_Z = Matrix(hvcat(((Nexo for _ in 1:Ntar)...,),
        (jacs.maps[v][t] for t in jacs.targets for v in exovars)...))
    ldiv!(lu!(H_U), H_Z)
    G_U = H_Z
    G_U .*= -one(eltype(G_U))
    Gs = IdDict{Symbol,IdDict{Symbol,Union{Matrix{TF},LinearMap{TF}}}}()
    for (j, z) in enumerate(exovars)
        Gs[z] = IdDict{Symbol,Union{Matrix{TF},LinearMap{TF}}}()
        for (i, u) in enumerate(endovars)
            Gs[z][u] = G_U[1+(i-1)*T:i*T, 1+(j-1)*T:j*T]
        end
    end
    return ImpulseResponseMaps(jacs, exovars, endovars, Gs)
end

function getG!(irm::ImpulseResponseMaps{TF}, exovar::Symbol, endovar::Symbol) where TF
    z = get(irm.Gs, exovar, nothing)
    z === nothing && throw(ArgumentError("$exovar is not an exogenous variable"))
    G = get(z, endovar, nothing)
    if G === nothing
        endovar in irm.jacs.vars ||
            throw(ArgumentError("$endovar is not an endogenous variable"))
        # M_U combines all indirect effects while M_u is for a specific channel
        M_U = LinearMap(UniformScaling(zero(TF)), irm.jacs.T)
        M_Z = M_U
        for (u, ms) in irm.jacs.maps
            if u === exovar
                M = get(ms, endovar, nothing)
                M === nothing || (M_Z = M)
            else
                M_u = get(ms, endovar, nothing)
                M_u === nothing && continue
                M_U += M_u * irm.Gs[exovar][u]
            end
        end
        G = Matrix(M_U + M_Z)
        irm.Gs[exovar][endovar] = G
    end
    return G
end


