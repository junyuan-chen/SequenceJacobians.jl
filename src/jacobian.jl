const DMap{TF} = Dict{Symbol, AbstractJacobianMap{TF}}

struct TotalJacobian{TF<:AbstractFloat, NT<:NamedTuple}
    parent::SequenceSpaceModel
    lookupblk::Dict{AbstractBlock, Int}
    dag::SimpleDiGraph{Int}
    pool::Vector{Symbol}
    invpool::Dict{Symbol,Int}
    varvals::NT
    nT::Int
    srcs::Vector{Symbol}
    tars::Vector{Symbol}
    ntarsrc::Vector{Int}
    excluded::Union{Set{BlockOrVar},Nothing}
    blkjacs::Vector{AbstractBlockJacobian}
    parts::Dict{Symbol, Dict{Symbol,VarJacobian{TF}}}
    totals::Dict{Symbol, Dict{Symbol,AbstractJacobianMap{TF}}}
end

function TotalJacobian(m::SequenceSpaceModel, sources, targets,
        varvals::NamedTuple, nT::Int; excluded=nothing, TF::Type=Float64)
    sources isa Symbol && (sources = (sources,))
    targets isa Symbol && (targets = (targets,))
    excluded isa BlockOrVar && (excluded = (excluded,))
    isempty(sources) && throw(ArgumentError("sources cannot be empty"))
    isempty(targets) && throw(ArgumentError("targets cannot be empty"))
    # Maintain the order for sources and targets
    sources = collect(Symbol, sources)
    targets = collect(Symbol, targets)
    invpool = Dict{Symbol,Int}(n=>i for (i, n) in enumerate(sources))
    any(v->haskey(invpool, v), targets) && throw(ArgumentError(
        "sources and targets cannot overlap"))
    pool = copy(sources)
    edges = Edge{Int}[]
    excluded === nothing || (excluded = Set{BlockOrVar}(excluded))
    lookupblk = Dict{AbstractBlock, Int}()
    blkjacs = AbstractBlockJacobian[]
    DVar = Dict{Symbol, VarJacobian{TF}}
    parts = Dict{Symbol, DVar}()
    totals = Dict{Symbol, DMap{TF}}(u=>DMap{TF}() for u in sources)
    for v in m.order
        isblock(m, v) || continue
        blk = m.pool[v]
        excluded !== nothing && blk in excluded && continue
        # Only need variables that are reachable from sources
        iins = [i for (i, vi) in enumerate(inputs(blk)) if haskey(invpool, vi)]
        length(iins) > 0 || continue
        # Compute all relevant Jacobians from the block
        J = jacobian(blk, iins, nT, varvals)
        push!(blkjacs, J)
        lookupblk[blk] = length(blkjacs)
        # Collect Jacobians by input/output variable and combine temporal terms
        for (r, vo) in enumerate(outputs(blk))
            excluded !== nothing && vo in excluded && continue
            push!(pool, vo)
            invpool[vo] = length(pool)
            jo = get!(DVar, parts, vo)
            for (ii, i) in enumerate(iins)
                vi = inputs(blk)[i]
                j0 = get(jo, vi, nothing)
                j = J[r,ii]
                if j0 === nothing
                    jo[vi] = j
                else # j0 exists if an input has leads/lags
                    # Only intended to be used for Shift
                    jo[vi] = j + j0
                end
            end
        end
        # Assemble maps for the total Jacobians
        # Jacobians from multiple temporal inputs are already combined
        for vi in unique!(map(i->inputs(blk)[i], iins))
            for vo in outputs(blk)
                excluded !== nothing && vo in excluded && continue
                push!(edges, Edge(invpool[vi], invpool[vo]))
                j = parts[vo][vi]
                # Handle the case when vi is a source of the DAG
                unknown = get(totals, vi, nothing)
                # If vi is not a source
                if unknown === nothing
                    # Iterate over total Jacobians for each source
                    for d in values(totals)
                        # Must have been reached from at least one source
                        maplast = get(d, vi, nothing)
                        if maplast !== nothing
                            # Combine possibly multiple channels
                            map0 = get(d, vo, nothing)
                            if map0 === nothing
                                d[vo] = j * maplast
                            else
                                muladd!(map0, j, maplast)
                            end
                        end
                    end
                # else vi is a source; combine possibly multiple channels
                else
                    map0 = get(unknown, vo, nothing)
                    if map0 === nothing
                        unknown[vo] = jacmap(j)
                    else
                        muladd!(map0, j)
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
    dag = SimpleDiGraphFromIterator(edges)
    return TotalJacobian(m, lookupblk, dag, pool, invpool, varvals, nT, sources,
        targets, ntarsrc, excluded, blkjacs, parts, totals)
end

inlength(j::TotalJacobian) = length(j.srcs)
tarlength(j::TotalJacobian) = length(j.tars)

getindex(j::TotalJacobian, src::Symbol) = j.totals[src]
getindex(j::TotalJacobian, src::Symbol, dest::Symbol) = j.totals[src][dest]

show(io::IO, j::TotalJacobian) =
    (print(io, "TotalJacobian("); join(io, j.tars, ", "); print(io, ')'))

function show(io::IO, ::MIME"text/plain", j::TotalJacobian{TF}) where TF
    nblk = length(j.blkjacs)
    nvar = length(j.pool)
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

struct GEJacobian{TF<:AbstractFloat, NT, ZB<:BlockMatrix, UB<:BlockMatrix}
    tjac::TotalJacobian{TF,NT}
    exovars::Vector{Symbol}
    endovars::Vector{Symbol}
    H_Z::Matrix{TF}
    H_Zblks::ZB
    iH_Zshift::Vector{Tuple{Int,Int}}
    H_U::Matrix{TF}
    H_Ublks::UB
    iH_Ushift::Vector{Tuple{Int,Int}}
    G_U::PseudoBlockMat{TF}
    nTfull::Int
end

function _fill_jac!(out, ishift, totals, vars, varwidths, tars, tarwidths, nT, TF)
    for j in axes(out, 2)
        v = vars[j]
        for i in axes(out, 1)
            vtar = tars[i]
            jac = get(totals[v], vtar, nothing)
            if jac === nothing
                out[i,j] = Zeros{TF}(nT*tarwidths[i], nT*varwidths[j])
            elseif jac isa MatrixMap
                out[i,j] = jac.out
            else
                out[i,j] = jac(nT)
                push!(ishift, (i,j))
            end
        end
    end
end

const ZMat{TF} = Zeros{TF, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}

function GEJacobian(tjac::TotalJacobian{TF}, exovars; nTfull::Int=tjac.nT) where TF
    tars = tjac.tars
    exovars isa Symbol && (exovars = (exovars,))
    isempty(exovars) && throw(ArgumentError("exovars cannot be empty"))
    for var in exovars
        var in tjac.srcs || throw(ArgumentError("$var is not a source variable"))
    end
    exovars = collect(Symbol, exovars)
    endovars = collect(setdiff(tjac.srcs, exovars))
    nT = tjac.nT
    vals = tjac.varvals
    ntar = length(tars)
    nendo = length(endovars)
    nexo = length(exovars)
    tarwidths = map(n->length(vals[n]), tars)
    endowidths = map(n->length(vals[n]), endovars)
    exowidths = map(n->length(vals[n]), exovars)
    H_Ublks = Matrix{Union{Matrix{TF},SubMat{TF},ZMat{TF}}}(undef, ntar, nendo)
    iH_Ushift = Tuple{Int,Int}[]
    _fill_jac!(H_Ublks, iH_Ushift, tjac.totals, endovars, endowidths, tars, tarwidths, nT, TF)
    H_Ublks = mortar(H_Ublks)
    H_Zblks = Matrix{Union{Matrix{TF},SubMat{TF},ZMat{TF}}}(undef, ntar, nexo)
    iH_Zshift = Tuple{Int,Int}[]
    _fill_jac!(H_Zblks, iH_Zshift, tjac.totals, exovars, exowidths, tars, tarwidths, nT, TF)
    H_Zblks = mortar(H_Zblks)
    H_U = Array(H_Ublks)
    H_Z = Array(H_Zblks)
    try
        H_U = lu!(H_U)
    catch e
        @warn e
        return H_Ublks, H_Zblks
    end
    G_U = similar(H_Z)
    ldiv!(G_U, H_U, H_Z)
    rmul!(G_U, -one(eltype(G_U)))
    G_U = PseudoBlockMatrix(G_U, nT*endowidths, nT*exowidths)
    return GEJacobian(tjac, exovars, endovars, H_Z, H_Zblks, iH_Zshift, H_U.factors,
        H_Ublks, iH_Ushift, G_U, nTfull)
end

show(io::IO, gj::GEJacobian) =
    (print(io, "GEJacobian("); join(io, gj.tjac.tars, ", "); print(io, ')'))

function show(io::IO, ::MIME"text/plain", j::GEJacobian{TF}) where TF
    print(io, "GEJacobian{$TF} with ", j.tjac.nT, " period")
    j.tjac.nT > 1 && print(io, 's')
    println(io, ":")
    print(io, "  exogenous:  ")
    join(io, j.exovars, ", ")
    print(io, "\n  endogenous: ")
    join(io, j.endovars, ", ")
    print(io, "\n  targets:    ")
    join(io, j.tjac.tars, ", ")
end

struct GEJacobianUpdatePlan{GJ, BJ<:Tuple}
    gj::GJ
    bjs::BJ
    smaps::Vector{ShiftMap}
    mmaps::Vector{Pair{Vector{Int},Vector{MatrixMap}}}
    itarH_Z::Vector{Tuple{Int,Int}}
    itarH_U::Vector{Tuple{Int,Int}}
end

function plan(gj::GEJacobian, params)
    tjac = gj.tjac
    params isa Symbol && (params = (params,))
    m = tjac.parent
    vtars = map(x->m.invpool[x], tjac.tars)
    vups = Set{Int}()
    for v in vtars
        union!(vups, dfs_parents(m.dag, v, dir=:in))
    end
    delete!(vups, 0)
    # Find vertices that can be reached from params
    vparams = map(x->m.invpool[x], params)
    vdowns = Set{Int}()
    for v in vparams
        haskey(tjac.invpool, v) && throw(ArgumentError(
            "$v can be directly reached from source variables"))
        vdown = dfs_parents(m.dag, v, dir=:out)
        union!(vdowns, vdown)
    end
    delete!(vdowns, 0)
    itar = Int[] # Record targets that are affected
    for (i, v) in enumerate(vtars)
        v in vdowns && push!(itar, i)
    end
    vsrcs = map(x->m.invpool[x], tjac.srcs)
    vreachedbysrc = [delete!(Set(dfs_parents(m.dag, v, dir=:out)), 0) for v in vsrcs]
    vaffected = intersect!(vdowns, vups)
    vblks = Int[]
    iblks = Int[] # Find blocks whose Jacobians need to be recomputed
    for v in m.order # The order matters
        v in vaffected || continue
        blk = m.pool[v]
        blk isa Symbol && continue
        any(x->v in x, vreachedbysrc) || continue
        push!(vblks, v)
        push!(iblks, tjac.lookupblk[blk])
    end
    smaps = ShiftMap[]
    # Determine which ins in MatrixMap may need to be updated
    mmaps = Pair{Vector{Int},Vector{MatrixMap}}[]
    for (s, vreached) in enumerate(vreachedbysrc)
        d = tjac.totals[tjac.srcs[s]]
        for v in vblks
            v in vreached || continue
            blk = m.pool[v]
            if blk isa SimpleBlock
                push!(smaps, (d[vo] for vo in outputs(blk))...)
            else
                # Order of ins needs to match the map in totals
                ins = intersect!(unique!([m.invpool[n] for n in inputs(blk)]), vreached)
                iinaffected = findall(in(vaffected), ins)
                mmapaffected = MatrixMap[d[vo] for vo in outputs(blk)]
                # ins for variables from the same block are identical
                # Share the same matrices for ins to avoid repetition
                # This is only relevant when the ins are for ShiftMaps
                mmap1ins = mmapaffected[1].ins
                for mmap in mmapaffected
                    copyto!(mmap.ins, mmap1ins)
                end
                push!(mmaps, iinaffected=>mmapaffected)
            end
        end
    end
    ibj = sort!(iblks) # The order matters
    bjs = ntuple(i->tjac.blkjacs[ibj[i]], length(ibj))
    f(x) = x[1] ∈ itar
    itarH_Z = filter(f, gj.iH_Zshift)
    itarH_U = filter(f, gj.iH_Ushift)
    return GEJacobianUpdatePlan(gj, bjs, smaps, mmaps, itarH_Z, itarH_U)
end

function _update_blkjacs!(p::GEJacobianUpdatePlan{GJ,BJ}, varvals::NamedTuple) where {GJ,BJ}
    if @generated
        ex = :()
        for i in 1:length(BJ.parameters)
            ex = :($ex; p.bjs[$i](varvals))
        end
        return ex
    else
        for j in p.bjs
            j(varvals)
        end
    end
end

function (p::GEJacobianUpdatePlan)(varvals::NamedTuple)
    gj = p.gj
    tjac = gj.tjac
    _update_blkjacs!(p, varvals)
    for smap in p.smaps
        _updateout!(smap)
    end
    for (iins, Mmaps) in p.mmaps
        # ins for the other Mmaps should have been handled due to the sharing of arrays
        _updateins!(Mmaps[1], iins)
        for Mmap in Mmaps
            _updateout!(Mmap)
        end
    end
    for (i, j) in p.itarH_Z
        Smap = tjac[gj.exovars[j]][tjac.tars[i]]
        mul!(view(gj.H_Zblks, Block(i,j)), Smap, true)
    end
    for (i, j) in p.itarH_U
        Smap = tjac[gj.endovars[j]][tjac.tars[i]]
        mul!(view(gj.H_Ublks, Block(i,j)), Smap, true)
    end
    copyto!(gj.H_Z, gj.H_Zblks)
    copyto!(gj.H_U, gj.H_Ublks)
    # lu! allocates
    H_U = lu!(gj.H_U)
    ldiv!(gj.G_U.blocks, H_U, gj.H_Z)
    rmul!(gj.G_U.blocks, -one(eltype(gj.G_U)))
    return p.gj
end

show(io::IO, p::GEJacobianUpdatePlan) =
    print(io, "GEJacobianUpdatePlan(", length(p.bjs), ")")

function show(io::IO, ::MIME"text/plain", p::GEJacobianUpdatePlan)
    print(io, "GEJacobianUpdatePlan with ", length(p.bjs), " block jacobian")
    print(io, length(p.bjs)>1 ? "s:" : ":")
    for j in p.bjs
        print(io, "\n  ", j)
    end
end

struct GMaps{TF, GJ<:GEJacobian{TF}} <: AbstractDict{Symbol, DMap{TF}}
    gj::GJ
    vreached::Vector{Int}
    Gs::Dict{Symbol, Dict{Symbol, AbstractJacobianMap{TF}}}
end

iterate(gs::GMaps) = iterate(gs.Gs)
iterate(gs::GMaps, state) = iterate(gs.Gs, state)
length(gs::GMaps) = length(gs.Gs)

function GMaps(gj::GEJacobian{TF}, endovars=nothing) where TF
    tjac = gj.tjac
    parts = gj.tjac.parts
    endovars isa Symbol && (endovars = (endovars,))
    if endovars !== nothing
        vreached = Set{Int}()
        # endo does not have to be a source but must have been encountered by tjac
        for endo in endovars
            v = get(tjac.invpool, endo, nothing)
            v === nothing && throw(ArgumentError(
                "$endo is not a variable reached from any source"))
            endo in gj.exovars && throw(ArgumentError("$endo is an exogenous variable"))
            union!(vreached, dfs_parents(tjac.dag, v, dir=:in))
        end
        # srcs are always at the beginning of pool
        setdiff!(vreached, 1:length(tjac.srcs))
        delete!(vreached, 0)
        vreached = sort!(collect(vreached))
    else # Consider all endogenous variables from TotalJacobian
        vreached = collect(length(tjac.srcs)+1:length(tjac.pool))
    end
    Gs = Dict{Symbol, DMap{TF}}()
    for (j, z) in enumerate(gj.exovars)
        Gs[z] = Gz = DMap{TF}()
        for (i, u) in enumerate(gj.endovars)
            Gz[u] = jacmap(view(gj.G_U, Block(i, j)))
        end
    end
    for v in vreached
        vo = tjac.pool[v]
        js = parts[vo]
        for vi in keys(js)
            exo = get(Gs, vi, nothing)
            if exo === nothing
                for d in values(Gs)
                    maplast = get(d, vi, nothing)
                    if maplast !== nothing
                        map0 = get(d, vo, nothing)
                        if map0 === nothing
                            d[vo] = js[vi] * maplast
                        else
                            muladd!(map0, js[vi], maplast)
                        end
                    end
                end
            else
                map0 = get(exo, vo, nothing)
                if map0 === nothing
                    exo[vo] = jacmap(js[vi])
                else
                    muladd!(map0, js[vi])
                end
            end
        end
    end
    return GMaps(gj, vreached, Gs)
end

@inline getindex(g::GMaps, exo::Symbol) = g.Gs[exo]

@inline function getindex(g::GMaps{TF}, exo::Symbol, endo::Symbol) where TF
    nT = g.gj.nTfull
    wi = length(g.gj.tjac.varvals[exo])
    wo = length(g.gj.tjac.varvals[endo])
    out = Matrix{TF}(undef, nT*wo, nT*wi)
    return mul!(out, g.Gs[exo][endo], true)
end

function (g::GMaps)()
    pool = g.gj.tjac.pool
    for z in g.gj.exovars
        Gz = g.Gs[z]
        for u in g.gj.endovars
            _updateout!(Gz[u])
        end
        for v in g.vreached
            Gzv = get(Gz, pool[v], nothing)
            Gzv === nothing && continue
            Gzv isa MatrixMap && _updateins!(Gzv)
            _updateout!(Gzv)
        end
    end
    return g
end

(g::GMaps)(out::AbstractMatrix, exo::Symbol, endo::Symbol) =
    mul!(out, g.Gs[exo][endo], true)

show(io::IO, gs::GMaps{TF}) where TF =
    (print(io, "GMaps{$TF}("); join(io, gs.gj.exovars, ", "); print(io, ')'))
