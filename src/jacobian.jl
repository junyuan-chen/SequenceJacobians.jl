struct TotalJacobian{TF<:AbstractFloat,NT<:NamedTuple}
    parent::SequenceSpaceModel
    blks::Vector{AbstractBlock}
    vars::Set{Symbol}
    varvals::NT
    nT::Int
    srcs::Set{Symbol}
    tars::Vector{Symbol}
    ntarsrc::Vector{Int}
    excluded::Union{Set{BlockOrVar},Nothing}
    parts::Dict{Symbol,Dict{Symbol,Matrix{LinearMap{TF}}}}
    totals::Dict{Symbol,Dict{Symbol,LinearMap{TF}}}
end

function TotalJacobian(m::SequenceSpaceModel, sources, targets,
        varvals::NamedTuple, nT::Int; excluded=nothing, TF::Type=Float64)
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
        # For some blocks, Jacobians for all inputs are obtained together
        byinput = jacbyinput(blk)
        byinput || (J = jacobian(blk, nT, varvals))
        # Compute direct Jacobians for each input-output pair
        for (i, vi) in enumerate(inputs(blk))
            # Only need variables that are reachable from sources
            if vi in vars
                byinput && (J = jacobian(blk, Val(i), nT, varvals))
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

inlength(j::TotalJacobian) = length(j.srcs)
tarlength(j::TotalJacobian) = length(j.tars)

show(io::IO, ::TotalJacobian{TF}) where TF = print(io, "TotalJacobian{$TF}")

function show(io::IO, ::MIME"text/plain", j::TotalJacobian{TF}) where TF
    nblk = length(j.blks)
    nvar = length(j.vars)
    print(io, "TotalJacobian{$TF} with $nblk block")
    nblk > 1 && print(io, 's')
    print(io, ", $nvar variable")
    nvar > 1 && print(io, 's')
    print(io, " and ", j.nT, " period")
    j.nT > 1 && print(io, 's')
    println(io, ":")
    print(io, "  sources: ")
    join(io, sort([j.srcs...]), ", ")
    print(io, "\n  targets: ")
    join(io, j.tars, ", ")
end

struct GEJacobian{TF<:AbstractFloat, HU<:Union{Matrix{TF},Nothing},
        FAC<:Union{<:Factorization,Nothing}}
    tjac::TotalJacobian{TF}
    exovars::Vector{Symbol}
    unknowns::Vector{Symbol}
    H_U::HU
    factor::FAC
    Gs::Dict{Symbol,Dict{Symbol,Matrix{TF}}}
end

function GEJacobian(tjac::TotalJacobian{TF}, exovars;
        keepH_U::Bool=false, keepfactor::Bool=false) where TF
    exovars isa Symbol && (exovars = (exovars,))
    isempty(exovars) && throw(ArgumentError("exovars cannot be empty"))
    for var in exovars
        var in tjac.srcs || throw(ArgumentError("$var is not a source variable"))
    end
    exovars = collect(Symbol, exovars)
    unknowns = collect(setdiff(tjac.srcs, exovars))
    nT = tjac.nT
    nZ = length(exovars)
    nU = length(unknowns)
    ntar = length(tjac.tars)
    zmap = LinearMap(UniformScaling(zero(TF)), nT)
    H_U = Matrix(hvcat(ntuple(i->nU, ntar),
        (get(tjac.totals[v], t, zmap) for t in tjac.tars for v in unknowns)...))
    H_Z = Matrix(hvcat(ntuple(i->nZ, ntar),
        (get(tjac.totals[v], t, zmap) for t in tjac.tars for v in exovars)...))
    keepH_U && (hu = copy(H_U))
    H_U = lu!(H_U)
    ldiv!(H_U, H_Z)
    G_U = H_Z
    G_U .*= -one(eltype(G_U))
    Gs = Dict{Symbol,Dict{Symbol,Matrix{TF}}}()
    j0 = 0
    for z in exovars
        Gs[z] = Dict{Symbol,Matrix{TF}}()
        Nz = length(tjac.varvals[z])
        i0 = 0
        for u in unknowns
            Nu = length(tjac.varvals[u])
            Gs[z][u] = G_U[i0+1:i0+Nu*nT, j0+1:j0+Nz*nT]
            i0 += Nu*nT
        end
        j0 += Nz*nT
    end
    return GEJacobian(tjac, exovars, unknowns,
        keepH_U ? hu : nothing, keepfactor ? H_U : nothing, Gs)
end

show(io::IO, ::GEJacobian{TF}) where TF = print(io, "GEJacobian{$TF}")

function show(io::IO, ::MIME"text/plain", j::GEJacobian{TF}) where TF
    print(io, "GEJacobian{$TF} with ", j.tjac.nT, " period")
    j.tjac.nT > 1 && print(io, 's')
    println(io, ":")
    print(io, "  exogenous:  ")
    join(io, j.exovars, ", ")
    print(io, "\n  endogenous: ")
    join(io, j.unknowns, ", ")
    print(io, "\n  targets:    ")
    join(io, j.tjac.tars, ", ")
end

function getG!(gejac::GEJacobian{TF}, exovar::Symbol, endovar::Symbol) where TF
    haskey(gejac.Gs, exovar) || throw(ArgumentError("$exovar is not an exogenous variable"))
    Gz = gejac.Gs[exovar]
    # G is readily available if endovar is a source
    haskey(Gz, endovar) && return Gz[endovar]
    # endovar does not have to be a source but must have been encountered by tjac
    endovar in gejac.tjac.vars ||
        throw(ArgumentError("$endovar is not an endogenous variable"))
    # M_U combines all indirect effects while M_u is for a specific channel
    M = LinearMap(UniformScaling(zero(TF)), gejac.tjac.nT)
    for (src, ms) in gejac.tjac.totals
        # Direct effect of exovar M_Z
        if src === exovar
            if haskey(ms, endovar)
                M += ms[endovar]
            end
        # Indirect effect via unknowns M_U
        elseif src in gejac.unknowns
            if haskey(ms, endovar) 
                M_u = ms[endovar]
                M += M_u * Gz[src]
            end
        end
    end
    G = Matrix(M)
    Gz[endovar] = G
    return G
end
