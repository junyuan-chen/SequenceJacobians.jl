const DMap{TF} = Dict{Symbol, AbstractJacobianMap{TF}}

struct TotalJacobian{TF<:AbstractFloat, NT<:NamedTuple}
    parent::SequenceSpaceModel
    lookupblk::Dict{AbstractBlock, Int}
    dag::SimpleDiGraph{Int}
    pool::Vector{Symbol}
    invpool::Dict{Symbol,Int}
    varvals::RefValue{NT}
    nT::Int
    ncol::Vector{Int}
    srcs::Vector{Symbol}
    tars::Vector{Symbol}
    nsrcbytar::Vector{Int}
    ntarbysrc::Vector{Int}
    excluded::Union{Set{BlockOrVar}, Nothing}
    blkjacs::Vector{AbstractBlockJacobian}
    dZs::Union{Dict{Symbol, Matrix{TF}}, Nothing}
    parts::Dict{Symbol, Dict{Symbol,Any}}
    totals::Dict{Symbol, Dict{Symbol,AbstractJacobianMap{TF}}}
end

function TotalJacobian(m::SequenceSpaceModel, sources, targets, varvals::NamedTuple, nT::Int;
        dZs=nothing, excluded=nothing, TF::Type=Float64)
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
    if dZs === nothing
        ncol = map(n->length(varvals[n]) * nT, sources)
    else
        dZs = Dict{Symbol, Matrix{TF}}(
            k=>reshape(collect(TF, v), size(v,1), size(v,2)) for (k, v) in dZs)
        ncol = [haskey(dZs, n) ? size(dZs[n], 2) : length(varvals[n]) * nT for n in sources]
    end
    excluded === nothing || (excluded = Set{BlockOrVar}(excluded))
    lookupblk = Dict{AbstractBlock, Int}()
    blkjacs = AbstractBlockJacobian[]
    DVar = Dict{Symbol, Any} # Do not restrict element type here
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
            # Handle the case when vi is a source separately
            unknown = get(totals, vi, nothing)
            # If vi is not a source
            if unknown === nothing
                # Iterate over total Jacobians for each source
                for d in values(totals)
                    # Must have been reached from at least one source
                    maplast = get(d, vi, nothing)
                    if maplast !== nothing
                        makeinmat = maplast isa ShiftMap && !(blk isa SimpleBlock)
                        inmat = makeinmat ? maplast(nT) : nothing
                        for vo in outputs(blk)
                            excluded !== nothing && vo in excluded && continue
                            push!(edges, Edge(invpool[vi], invpool[vo]))
                            j = parts[vo][vi]
                            # Combine possibly multiple channels
                            map0 = get(d, vo, nothing)
                            if map0 === nothing
                                d[vo] = jacmap(j, maplast, inmat)
                            else
                                muladd!(map0, j, maplast, inmat)
                            end
                        end
                    end
                end
            else # vi is a source; combine possibly multiple channels
                dZ = dZs === nothing ? true : get(dZs, vi, true)
                for vo in outputs(blk)
                    excluded !== nothing && vo in excluded && continue
                    push!(edges, Edge(invpool[vi], invpool[vo]))
                    j = parts[vo][vi]
                    map0 = get(unknown, vo, nothing)
                    if map0 === nothing
                        unknown[vo] = jacmap(j, dZ)
                    else
                        muladd!(map0, j, dZ)
                    end
                end
            end
        end
    end
    # Record the number of sources each target can be reached from
    nsrcbytar = zeros(Int, length(targets))
    ntarbysrc = zeros(Int, length(sources))
    for (j, n) in enumerate(sources)
        d = totals[n]
        for (i, v) in enumerate(targets)
            if haskey(d, v)
                nsrcbytar[i] += 1
                ntarbysrc[j] += 1
            end
        end
    end
    nrt = targets[nsrcbytar.<2]
    length(nrt) > 0 && @warn "not all targets are reachable from at least two sources; check "*join(nrt, ", ")
    nrs = sources[ntarbysrc.<1]
    length(nrs) > 0 && @info "not all sources reach at least one target; check "*join(nrs, ", ")
    dag = SimpleDiGraphFromIterator(edges)
    return TotalJacobian(m, lookupblk, dag, pool, invpool, Ref(varvals), nT, ncol,
        sources, targets, nsrcbytar, ntarbysrc, excluded, blkjacs, dZs, parts, totals)
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

const BlockMat{TF} = BlockMatrix{TF, Matrix{Union{Zeros{TF, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, SubArray{TF, 2, Matrix{TF}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}, Matrix{TF}}}, Tuple{BlockedUnitRange{Vector{Int64}}, BlockedUnitRange{Vector{Int64}}}}

struct GEJacobian{TF<:AbstractFloat, NT}
    tjac::TotalJacobian{TF,NT}
    exovars::Vector{Symbol}
    endosrcs::Vector{Symbol}
    H_Z::PseudoBlockMat{TF}
    H_Zblks::BlockMat{TF}
    iH_Zshift::Vector{Tuple{Int,Int}}
    H_Uws::LUWs
    H_U::PseudoBlockMat{TF}
    H_Ublks::BlockMat{TF}
    iH_Ushift::Vector{Tuple{Int,Int}}
    G_U::PseudoBlockMat{TF}
    nTfull::Int
    nZcol::Vector{Int}
end

function _fill_jac!(out, ishift, totals, vars, varwidths, tars, tarwidths, nT, ncol, TF)
    for j in axes(out, 2)
        nC = ncol === nothing ? nT*varwidths[j] : ncol[j]
        v = vars[j]
        for i in axes(out, 1)
            vtar = tars[i]
            jac = get(totals[v], vtar, nothing)
            if jac === nothing
                out[i,j] = Zeros{TF}(nT*tarwidths[i], nC)
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

function GEJacobian(tjac::TotalJacobian{TF}, exovars, endosrcs=nothing;
        nTfull::Int=tjac.nT) where TF
    tars = tjac.tars
    nT = tjac.nT
    exovars isa Symbol && (exovars = (exovars,))
    isempty(exovars) && throw(ArgumentError("exovars cannot be empty"))
    exovars = collect(Symbol, exovars)
    vals = tjac.varvals[]
    nZcol = Int[]
    for var in exovars
        i = findfirst(==(var), tjac.srcs)
        i === nothing && throw(ArgumentError("$var is not a source variable"))
        push!(nZcol, tjac.ncol[i])
    end
    if endosrcs === nothing
        iendo = findall(!in(exovars), tjac.srcs)
        for i in iendo
            tjac.ncol[i] == nT*length(vals[tjac.srcs[i]]) || throw(ArgumentError(
                "Jacobians for endogenous source $(tjac.srcs[i]) must be square; check the dZs option with TotalJacobian"))
        end
        endosrcs = tjac.srcs[iendo]
    else
        for var in endosrcs
            i = findfirst(==(var), tjac.srcs)
            i === nothing && throw(ArgumentError("$var is not a source variable"))
            var in exovars && throw(ArgumentError("$var is exogenous"))
            tjac.ncol[i] == nT*length(vals[tjac.srcs[i]]) || throw(ArgumentError(
                "Jacobians for endogenous source $(tjac.srcs[i]) must be square; check the dZs option with TotalJacobian"))
        end
    end
    ntar = length(tars)
    nendo = length(endosrcs)
    nexo = length(exovars)
    tarwidths = map(n->length(vals[n]), tars)
    endowidths = map(n->length(vals[n]), endosrcs)
    exowidths = map(n->length(vals[n]), exovars)
    H_Ublks = Matrix{Union{Matrix{TF},SubMat{TF},ZMat{TF}}}(undef, ntar, nendo)
    iH_Ushift = Tuple{Int,Int}[]
    _fill_jac!(H_Ublks, iH_Ushift, tjac.totals, endosrcs, endowidths, tars, tarwidths,
        nT, nothing, TF)
    H_Ublks = mortar(H_Ublks)
    H_Zblks = Matrix{Union{Matrix{TF},SubMat{TF},ZMat{TF}}}(undef, ntar, nexo)
    iH_Zshift = Tuple{Int,Int}[]
    _fill_jac!(H_Zblks, iH_Zshift, tjac.totals, exovars, exowidths, tars, tarwidths,
        nT, nZcol, TF)
    H_Zblks = mortar(H_Zblks)
    H_U = PseudoBlockMatrix(Array(H_Ublks), axes(H_Ublks))
    H_Z = PseudoBlockMatrix(Array(H_Zblks), axes(H_Zblks))
    ws = LUWs(H_U.blocks)
    H_Ulu = LU(LAPACK.getrf!(ws, H_U.blocks)...)
    G_U = similar(H_Z)
    ldiv!(G_U, H_Ulu, H_Z)
    rmul!(G_U, -one(eltype(G_U)))
    G_U = PseudoBlockMatrix(G_U, nT*endowidths, nZcol)
    return GEJacobian(tjac, exovars, endosrcs, H_Z, H_Zblks, iH_Zshift, ws, H_U,
        H_Ublks, iH_Ushift, G_U, nTfull, nZcol)
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
    join(io, j.endosrcs, ", ")
    print(io, "\n  targets:    ")
    join(io, j.tjac.tars, ", ")
end

struct GEJacobianUpdatePlan{TF, GJ<:GEJacobian{TF}, BJ<:Tuple}
    gj::GJ
    bjs::BJ
    smaps::Vector{ShiftMap{TF}}
    mmaps::Vector{Pair{Vector{Int},Vector{MatrixMap{TF}}}}
    itarH_Z::Vector{Tuple{Int,Int}}
    itarH_U::Vector{Tuple{Int,Int}}
end

function plan(gj::GEJacobian{TF}, params; dZvars=nothing) where TF
    tjac = gj.tjac
    lookupblk = tjac.lookupblk
    params isa Symbol && (params = (params,))
    m = tjac.parent
    vtars = map(x->m.invpool[x], tjac.tars)
    vups = Set{Int}()
    for v in vtars
        union!(vups, neighborhood(m.dag, v, nv(m.dag), dir=:in))
    end
    # Find vertices that can be reached from params
    vparams = map(x->m.invpool[x], params)
    vdowns = Set{Int}()
    # Find blocks whose Jacobians need to be recomputed
    idirectblks = Set{Int}() # indices for blkjacs
    for (v, n) in zip(vparams, params)
        haskey(tjac.invpool, v) && throw(ArgumentError(
            "$n can be directly reached from source variables"))
        vdown = neighborhood(m.dag, v, nv(m.dag), dir=:out)
        union!(vdowns, vdown)
        nreached = 0
        for vb in outneighbors(m.dag, v)
            blk = m.pool[vb]
            iblk = get(lookupblk, blk, nothing)
            if iblk !== nothing
                nreached += 1
                push!(idirectblks, iblk)
            end
        end
        nreached > 0 || throw(ArgumentError(
            "value of $n is irrelevant to any Jacobian involved in gj"))
    end
    itar = Int[] # Record targets that are affected
    for (i, v) in enumerate(vtars)
        v in vdowns && push!(itar, i)
    end
    vsrcs = map(x->m.invpool[x], tjac.srcs)
    vreachedbysrc = [neighborhood(m.dag, v, nv(m.dag), dir=:out) for v in vsrcs]
    vaffected = intersect!(vdowns, vups)
    smaps = ShiftMap{TF}[]
    # Determine which ins in MatrixMap may need to be updated
    mmaps = Pair{Vector{Int},Vector{MatrixMap{TF}}}[]
    for (s, vreached) in enumerate(vreachedbysrc)
        d = tjac.totals[tjac.srcs[s]]
        updateall = dZvars !== nothing && m.pool[vsrcs[s]] in dZvars
        for v in m.order # The order matters
            v in vreached || continue
            blk = m.pool[v]
            blk isa Symbol && continue
            v in vaffected || updateall || continue
            if blk isa SimpleBlock
                push!(smaps, (d[vo] for vo in outputs(blk))...)
            else
                if updateall
                    iinaffected = collect(1:length(d[outputs(blk)[1]].ins))
                else
                    bj = tjac.blkjacs[lookupblk[blk]]
                    # Order of ins needs to match the map in totals
                    ins = intersect!(unique!(map(i->m.invpool[inputs(blk)[i]], bj.iins)), vreached)
                    iinaffected = findall(in(vaffected), ins)
                end
                mmapaffected = MatrixMap{TF}[d[vo] for vo in outputs(blk)]
                # ins for variables from the same block are identical
                push!(mmaps, iinaffected=>mmapaffected)
            end
        end
    end
    ibj = sort!(collect(idirectblks)) # The order matters
    bjs = ntuple(i->tjac.blkjacs[ibj[i]], length(ibj))
    f(x) = x[1] ∈ itar
    itarH_Z = dZvars === nothing ? filter(f, gj.iH_Zshift) : gj.iH_Zshift
    itarH_U = filter(f, gj.iH_Ushift)
    return GEJacobianUpdatePlan(gj, bjs, smaps, mmaps, itarH_Z, itarH_U)
end

@inline function _update_blkjacs!(p::GEJacobianUpdatePlan{TF,GJ,BJ}, varvals::NamedTuple) where {TF,GJ,BJ}
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

@inline function (p::GEJacobianUpdatePlan)(varvals::NamedTuple)
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
        mul!(gj.H_Zblks.blocks[i,j], Smap, true)
    end
    for (i, j) in p.itarH_U
        Smap = tjac[gj.endosrcs[j]][tjac.tars[i]]
        mul!(gj.H_Ublks.blocks[i,j], Smap, true)
    end
    # This avoids 1 allocation from copyto!
    _copyto!(MemoryLayout(gj.H_Z), MemoryLayout(gj.H_Zblks), gj.H_Z, gj.H_Zblks)
    _copyto!(MemoryLayout(gj.H_U), MemoryLayout(gj.H_Ublks), gj.H_U, gj.H_Ublks)
    # Non-allocating alternative to lu!
    H_U = LU(LAPACK.getrf!(gj.H_Uws, gj.H_U.blocks)...)
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
    G_Umaps::Vector{MatrixMap{TF}}
    smaps::Vector{ShiftMap{TF}}
    mmaps::Vector{Vector{MatrixMap{TF}}}
    Gs::Dict{Symbol, Dict{Symbol, AbstractJacobianMap{TF}}}
    inds::Dict{Symbol, Dict{Symbol, Tuple{Int,Int}}}
end

function GMaps(gj::GEJacobian{TF}, endovars=nothing) where TF
    tjac = gj.tjac
    lookupblk = tjac.lookupblk
    m = tjac.parent
    nT = gj.nTfull
    parts = gj.tjac.parts
    endovars isa Symbol && (endovars = (endovars,))
    if endovars !== nothing
        vars = Set{Symbol}()
        iblks = Set{Int}()
        # endo does not have to be a source but must have been encountered by tjac
        for endo in endovars
            v = get(m.invpool, endo, nothing)
            v === nothing && throw(ArgumentError(
                "$endo is not a variable reached from any source"))
            endo in gj.exovars && throw(ArgumentError("$endo is an exogenous variable"))
            vup = neighborhood(m.dag, v, nv(m.dag), dir=:in)
            for v in vup
                obj = m.pool[v]
                obj isa Symbol ? push!(vars, obj) : push!(iblks, lookupblk[obj])
            end
        end
        iblks = sort!(collect(iblks))
    else # Consider all endogenous variables from TotalJacobian
        vars = nothing
        iblks = 1:length(tjac.blkjacs)
    end
    Gs = Dict{Symbol, DMap{TF}}()
    inds = Dict{Symbol, Dict{Symbol, Tuple{Int,Int}}}()
    G_Umaps = MatrixMap{TF}[]
    for (j, z) in enumerate(gj.exovars)
        Gs[z] = Gz = DMap{TF}()
        inds[z] = idz = Dict{Symbol, Tuple{Int,Int}}()
        for (i, u) in enumerate(gj.endosrcs)
            Gz[u] = jm = jacmap(view(gj.G_U, Block(i, j)))
            push!(G_Umaps, jm)
            idz[u] = (0, length(G_Umaps))
        end
    end
    smaps = ShiftMap{TF}[]
    mmaps = Vector{MatrixMap{TF}}[]
    for iblk in iblks
        bj = tjac.blkjacs[iblk]
        blk = bj.blk
        for vi in unique(inputs(blk))
            exo = get(Gs, vi, nothing)
            if exo === nothing
                for (z, d) in Gs
                    idz = inds[z]
                    maplast = get(d, vi, nothing)
                    if maplast !== nothing
                        # Need to collect mmaps with different ins separately
                        blk isa SimpleBlock ? (blkmmaps = nothing) :
                            (blkmmaps = MatrixMap{TF}[]; push!(mmaps, blkmmaps))
                        makeinmat = maplast isa ShiftMap && !(blk isa SimpleBlock)
                        inmat = makeinmat ? maplast(nT) : nothing
                        for vo in outputs(blk)
                            vars === nothing || vo in vars || continue
                            js = get(parts, vo, nothing)
                            js === nothing && continue
                            map0 = get(d, vo, nothing)
                            if map0 === nothing
                                d[vo] = jm = jacmap(js[vi], maplast, inmat)
                                if blkmmaps === nothing
                                    push!(smaps, jm)
                                    idz[vo] = (1, length(smaps))
                                else
                                    push!(blkmmaps, jm)
                                    idz[vo] = (1+length(mmaps), length(blkmmaps))
                                end
                            else
                                muladd!(map0, js[vi], maplast, inmat)
                            end
                        end
                    end
                end
            else # vi is an exogenous variable
                dZ = tjac.dZs === nothing ? true : get(tjac.dZs, vi, true)
                idz = inds[vi]
                blk isa SimpleBlock ? (blkmmaps = nothing) :
                    (blkmmaps = MatrixMap{TF}[]; push!(mmaps, blkmmaps))
                for vo in outputs(blk)
                    vars === nothing || vo in vars || continue
                    js = get(parts, vo, nothing)
                    js === nothing && continue
                    map0 = get(exo, vo, nothing)
                    if map0 === nothing
                        exo[vo] = jm = jacmap(js[vi], dZ)
                        if blkmmaps === nothing
                            push!(smaps, jm)
                            idz[vo] = (1, length(smaps))
                        else
                            push!(blkmmaps, jm)
                            idz[vo] = (1+length(mmaps), length(blkmmaps))
                        end
                    else
                        muladd!(map0, js[vi], dZ)
                    end
                end
            end
        end
    end
    return GMaps(gj, G_Umaps, smaps, mmaps, Gs, inds)
end

@inline getindex(g::GMaps{TF}, exo::Symbol) where TF = g.Gs[exo]

iterate(g::GMaps) = iterate(g.Gs)
iterate(g::GMaps, state) = iterate(g.Gs, state)
length(g::GMaps) = length(g.Gs)

@inline function getindex(g::GMaps, exo::Symbol, endo::Symbol)
    i, j = g.inds[exo][endo]
    if i === 0
        return g.G_Umaps[j]
    elseif i === 1
        return g.smaps[j]
    else
        return g.mmaps[i-1][j]
    end
end

@inline function (g::GMaps)()
    for m in g.G_Umaps
        _updateout!(m)
    end
    for m in g.smaps
        _updateout!(m)
    end
    for ms in g.mmaps
        isempty(ms) && continue
        _updateins!(ms[1])
        for m in ms
            _updateout!(m)
        end
    end
    return g
end

@inline function (g::GMaps)(out::AbstractVecOrMat, exo::Symbol, endo::Symbol)
    varvals = g.gj.tjac.varvals[]
    i, j = g.inds[exo][endo]
    if i === 0
        return mul!(out, g.G_Umaps[j], true; mb=length(varvals[endo]), nb=length(varvals[exo]))
    elseif i === 1
        return mul!(out, g.smaps[j], true)
    else
        return mul!(out, g.mmaps[i-1][j], true;
            mb=length(varvals[endo]), nb=length(varvals[exo]))
    end
end

@inline function (g::GMaps)(exo::Symbol, endo::Symbol)
    nT = g.gj.nTfull
    i, j = g.inds[exo][endo]
    if i === 0
        return copy(g.G_Umaps[j].out)
    elseif i === 1
        return g.smaps[j](nT)
    else
        return copy(g.mmaps[i-1][j].out)
    end
end

show(io::IO, gs::GMaps{TF}) where TF =
    (print(io, "GMaps{$TF}("); join(io, gs.gj.exovars, ", "); print(io, ')'))
