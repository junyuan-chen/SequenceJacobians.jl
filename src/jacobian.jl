const JacType{TF<:AbstractFloat} = Union{Matrix{TF},LinearMap{TF}}

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
    parts::Dict{Symbol,Dict{Symbol,Matrix{LinearMap{TF}}}}
    totals::Dict{Symbol,Dict{Symbol,LinearMap{TF}}}
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
    DMat = Dict{Symbol,Matrix{LinearMap{TF}}}
    parts = Dict{Symbol,DMat}()
    DMap = Dict{Symbol,LinearMap{TF}}
    totals = Dict{Symbol,DMap}(u=>DMap() for u in vars)
    zmap = LinearMap(UniformScaling(zero(TF)), nT)
    for v in m.order
        isblock(m, v) || continue
        blk = m.pool[v]
        excluded !== nothing && blk in excluded && continue
        pushed = false
        # Compute direct Jacobians for each input-output pair
        for (i, vi) in enumerate(inputs(blk))
            # Only need variables that are reachable from sources
            if vi in vars
                J = jacobian(blk, i, varvals)
                r0next = 0
                for (r, vo) in enumerate(outputs(blk))
                    # Input/output variable could be an array
                    Ni = length(varvals[vi])
                    No = outlength(blk, r)
                    # Make sure r0 is updated even if the iteration is skipped
                    r0 = r0next
                    r0next += No
                    excluded !== nothing && vo in excluded && continue
                    jo = get!(DMat, parts, vo)
                    mj = get(jo, vi, nothing)
                    for ii in 1:Ni
                        for rr in 1:No
                            j, isz = getjacmap(blk, J, i, ii, r, r0+rr, nT)
                            isz && continue
                            # Create the array mj only when nonzero map is encountered
                            if mj === nothing
                                mj = Matrix{LinearMap{TF}}(undef, No, Ni)
                                fill!(mj, zmap)
                                jo[vi] = mj
                            end
                            # j0 might be nonzero if multiple temporal inputs exist
                            j0 = mj[rr,ii]
                            # Replace zmap so that it is not accumulated
                            mj[rr,ii] = j0 === zmap ? j : j0+j
                        end
                    end
                    # Do not bring zeros into total Jacobians
                    mj === nothing && continue
                    push!(vars, vo)
                    if !pushed
                        push!(blks, blk)
                        pushed = true
                    end
                end
            end
        end
        # Assemble maps for the total Jacobians
        # Jacobians from multiple temporal inputs are already combined
        for vi in unique(inputs(blk))
            if vi in vars
                for vo in outputs(blk)
                    excluded !== nothing && vo in excluded && continue
                    mj = get(parts[vo], vi, nothing)
                    mj === nothing && continue
                    No, Ni = size(mj)
                    if Ni == 1 && No == 1
                        map = mj[1]
                    else
                        map = hvcat(((Ni for _ in 1:No)...,),
                            (mj[rr,ii] for rr in 1:No for ii in 1:Ni)...)
                    end
                    # Handle the case when vi is a source of the DAG
                    unknown = get(totals, vi, nothing)
                    # If vi is not a source
                    if unknown === nothing
                        # Iterate over total Jacobians for each source
                        for d in values(totals)
                            maplast = get(d, vi, nothing)
                            if maplast !== nothing
                                # Maps from multiple temporal terms are already summed in mj
                                mcomp = map * maplast
                                # Combine possibly multiple channels
                                map0 = get(d, vo, nothing)
                                d[vo] = map0 === nothing ? mcomp : mcomp + map0
                            end
                        end
                    # If vi is a source
                    else
                        # Combine possibly multiple channels
                        map0 = get(unknown, vo, nothing)
                        unknown[vo] = map0 === nothing ? map : map + map0
                    end
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
    any(ntarsrc.<2) && @warn "not all targets are reachable from at least two sources"
    return TotalJacobian(m, blks, vars, varvals, nT, sources, targets, ntarsrc, excluded, parts, totals)
end

struct GEJacobian{TF<:AbstractFloat}
    jacs::TotalJacobian{TF}
    exovars::Vector{Symbol}
    unknowns::Vector{Symbol}
    H_U::Union{Matrix{TF}, Nothing}
    factor::Union{LU{TF, Matrix{TF}},Nothing}
    Gs::Dict{Symbol,Dict{Symbol,JacType{TF}}}
end

function GEJacobian(jacs::TotalJacobian{TF}, exovars;
        keepH_U::Bool=false, keepfactor::Bool=false) where TF
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
    keepH_U && (hu = copy(H_U))
    H_U = lu!(H_U)
    ldiv!(H_U, H_Z)
    G_U = H_Z
    G_U .*= -one(eltype(G_U))
    Gs = Dict{Symbol,Dict{Symbol,JacType{TF}}}()
    j0 = 0
    for z in exovars
        Gs[z] = Dict{Symbol,JacType{TF}}()
        Nz = length(jacs.varvals[z])
        i0 = 0
        for u in unknowns
            Nu = length(jacs.varvals[u])
            Gs[z][u] = G_U[i0+1:i0+Nu*nT, j0+1:j0+Nz*nT]
            i0 += Nu*nT
        end
        j0 += Nz*nT
    end
    return GEJacobian(jacs, exovars, unknowns,
        keepH_U ? hu : nothing, keepfactor ? H_U : nothing, Gs)
end

function getG!(gejac::GEJacobian{TF}, exovar::Symbol, endovar::Symbol) where TF
    z = get(gejac.Gs, exovar, nothing)
    z === nothing && throw(ArgumentError("$exovar is not an exogenous variable"))
    # G is readily available if endovar is a source
    G = get(z, endovar, nothing)
    if G === nothing
        # endovar does not have to be a source
        endovar in gejac.jacs.vars ||
            throw(ArgumentError("$endovar is not an endogenous variable"))
        # M_U combines all indirect effects while M_u is for a specific channel
        M_U = LinearMap(UniformScaling(zero(TF)), gejac.jacs.nT)
        M_Z = M_U
        for (u, ms) in gejac.jacs.totals
            # Direct effect of exovar
            if u === exovar
                M = get(ms, endovar, nothing)
                M === nothing || (M_Z = M)
            # Indirect effect via other sources
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
