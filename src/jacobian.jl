const JacType{TF<:AbstractFloat} = LinearMap{TF}

struct TotalJacobian{TF<:AbstractFloat}
    parent::SequenceSpaceModel
    blks::Vector{AbstractBlock}
    vars::Set{Symbol}
    varvals::Dict{Symbol,ValType{TF}}
    nT::Int
    srcs::Set{Symbol}
    tars::Vector{Symbol}
    ntarsrc::Vector{Int}
    excluded::Union{Set{BlockOrVar},Nothing}
    parts::IdDict{Symbol,Vector{Pair{Symbol,JacType{TF}}}}
    totals::IdDict{Symbol,IdDict{Symbol,LinearMap}}
end

function TotalJacobian(m::SequenceSpaceModel, sources, targets,
        varvals::Dict{Symbol,ValType{TF}}, nT::Int, excluded=nothing) where TF
    sources isa Symbol && (sources = (sources,))
    targets isa Symbol && (targets = (targets,))
    excluded isa BlockOrVar && (excluded = (excluded,))
    isempty(sources) && throw(ArgumentError("sources cannot be empty"))
    isempty(targets) && throw(ArgumentError("targets cannot be empty"))
    sources = Set{Symbol}(sources)
    # The order of the targets need to be maintained
    targets = collect(Symbol, targets)
    any(v->v in sources, targets) && throw(ArgumentError("sources and targets cannot overlap"))
    excluded === nothing || (excluded = Set{BlockOrVar}(excluded))
    vars = copy(sources)
    blks = AbstractBlock[]
    parts = IdDict{Symbol,Vector{Pair{Symbol,JacType{TF}}}}()
    Dmap = IdDict{Symbol,LinearMap}
    totals = IdDict{Symbol,Dmap}(u=>Dmap() for u in vars)
    for v in m.order
        isblock(m, v) || continue
        blk = m.pool[v]
        excluded !== nothing && blk in excluded && continue
        pushed = false
        for (i, vi) in enumerate(inputs(blk))
            if vi in vars
                J = jacobian(blk, i, varvals)
                # Track the first row in Jacobian matrix in case vo is an array
                ir = 1
                for (r, vo) in enumerate(outputs(blk))
                    excluded !== nothing && vo in excluded && continue
                    js = get!(valtype(parts), parts, vo)
                    j, isz = getjacmap(blk, J, i, r, ir, nT)
                    ir += Int(size(j, 1) / nT)
                    push!(js, vi=>j)
                    # Do not proceed from zeros
                    isz && continue
                    push!(vars, vo)
                    # Handle the case when vi is a source of the DAG
                    unknown = get(totals, vi, nothing)
                    # If vi is not a source
                    if unknown === nothing
                        # Iterate over total Jacobians for each source
                        for d in values(totals)
                            maplast = get(d, vi, nothing)
                            if maplast !== nothing
                                jcomp = j * maplast
                                # vo may exist when temporal terms are involved
                                map = get(d, vo, nothing)
                                d[vo] = map === nothing ? jcomp : map+jcomp
                            end
                        end
                    # If vi is a source
                    else
                        # vo may exist when temporal terms are involved
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
    # Record the number of sources each target can be reached from
    ntarsrc = zeros(Int, length(targets))
    for d in values(totals)
        for (i, v) in enumerate(targets)
            if haskey(d, v)
                ntarsrc[i] += 1
            end
        end
    end
    any(ntarsrc.<1) &&
        @warn "not all targets are reachable from at least two sources"
    return TotalJacobian(m, blks, vars, varvals, nT, sources, targets, ntarsrc, excluded, parts, totals)
end

struct GEJacobian{TF<:AbstractFloat}
    jacs::TotalJacobian{TF}
    exovars::Vector{Symbol}
    unknowns::Vector{Symbol}
    H_U::Union{LU{TF, Matrix{TF}},Nothing}
    Gs::IdDict{Symbol,IdDict{Symbol,Union{Matrix{TF},LinearMap{TF}}}}
end

function GEJacobian(jacs::TotalJacobian{TF}, exovars; keepH_U::Bool=false) where TF
    exovars isa Symbol && (exovars = (exovars,))
    isempty(exovars) && throw(ArgumentError("exovars cannot be empty"))
    for var in exovars
        var in jacs.srcs || throw(ArgumentError("$var is not a source variable"))
    end
    exovars = collect(Symbol, exovars)
    unknowns = collect(setdiff(jacs.srcs, exovars))
    nT = jacs.nT
    nZ = length(exovars)
    nU = length(unknowns)
    ntar = length(jacs.tars)
    zmap = LinearMap(UniformScaling(zero(TF)), nT)
    H_U = Matrix(hvcat(((nU for _ in 1:ntar)...,),
        (get(jacs.totals[v], t, zmap) for t in jacs.tars for v in unknowns)...))
    H_Z = Matrix(hvcat(((nZ for _ in 1:ntar)...,),
        (get(jacs.totals[v], t, zmap) for t in jacs.tars for v in exovars)...))
    H_U = lu!(H_U)
    ldiv!(H_U, H_Z)
    G_U = H_Z
    G_U .*= -one(eltype(G_U))
    Gs = IdDict{Symbol,IdDict{Symbol,Union{Matrix{TF},LinearMap{TF}}}}()
    for (j, z) in enumerate(exovars)
        Gs[z] = IdDict{Symbol,Union{Matrix{TF},LinearMap{TF}}}()
        for (i, u) in enumerate(unknowns)
            Gs[z][u] = G_U[1+(i-1)*nT:i*nT, 1+(j-1)*nT:j*nT]
        end
    end
    return GEJacobian(jacs, exovars, unknowns, keepH_U ? H_U : nothing, Gs)
end

function getG!(gejac::GEJacobian{TF}, exovar::Symbol, endovar::Symbol) where TF
    z = get(gejac.Gs, exovar, nothing)
    z === nothing && throw(ArgumentError("$exovar is not an exogenous variable"))
    G = get(z, endovar, nothing)
    if G === nothing
        # endovar does not have to be a source
        endovar in gejac.jacs.vars ||
            throw(ArgumentError("$endovar is not an endogenous variable"))
        # M_U combines all indirect effects while M_u is for a specific channel
        M_U = LinearMap(UniformScaling(zero(TF)), gejac.jacs.nT)
        M_Z = M_U
        for (u, ms) in gejac.jacs.totals
            if u === exovar
                M = get(ms, endovar, nothing)
                M === nothing || (M_Z = M)
            else
                M_u = get(ms, endovar, nothing)
                M_u === nothing && continue
                M_U += M_u * gejac.Gs[exovar][u]
            end
        end
        G = Matrix(M_U + M_Z)
        gejac.Gs[exovar][endovar] = G
    end
    return G
end
