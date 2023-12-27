rootsolvercache(::Any, ::SteadyState; kwargs...) = nothing

struct CombinedBlock{NJ, ST, SS<:SteadyState, CA, LS, ins, outs} <: AbstractBlock{ins,outs}
    ssins::Set{Symbol}
    ss::SS
    sscache::RefValue{CA}
    ssargs::Dict{Symbol,Any}
    jacus::NTuple{NJ,Symbol}
    jactars::NTuple{NJ,Symbol}
    jacargs::Dict{Symbol,Any}
    static::Bool
    sparseH_U::Bool
    function CombinedBlock(ins::NTuple{NI,Symbol}, ssins::Set{Symbol},
            outs::NTuple{NO,Symbol}, ss::SS, solver, ssargs::Dict{Symbol,Any},
            jacus::NTuple{NJ,Symbol}, jactars::NTuple{NJ,Symbol},
            jacargs::Dict{Symbol,Any}, static::Bool, sparseH_U::Bool,
            LS::Type{<:AbstractLinearSolver}) where {NI,NO,SS<:SteadyState,NJ}
        if isrootsolver(solver)
            ST = solver isa Type ? solver : typeof(solver)
            sscache = rootsolvercache(ST, ss; ssargs...)
        elseif isrootsolvercache(solver)
            ST = typeof(solver)
            sscache = solver
        elseif solver === NoRootSolver
            ST = NoRootSolver
            sscache = nothing
        else
            throw(ArgumentError("solver is not recognized"))
        end
        m = model(ss)
        # No duplicate is allowed for extracting varvals with ins
        ins = (unique(ins)...,)
        if ST === NoRootSolver
            hastarget(ss) && throw(ArgumentError(
                "NoRootSolver is not allowed with nonempty steady state targets"))
            NJ > 0 && throw(ArgumentError(
                "NoRootSolver is not allowed with nonempty Jacobian targets"))
        else
            hastarget(ss) || throw(ArgumentError(
                "solver type must be NoRootSolver with empty steady state target"))
        end
        length(ins) > 0 || throw(ArgumentError("the inputs of a block cannot be empty"))
        length(outs) > 0 || throw(ArgumentError("the outputs of a block cannot be empty"))
        for vi in ins
            v = get(m.invpool, vi, 0)
            v === 0 && throw(ArgumentError("$vi is not a variable of the model"))
            if hastarget(ss)
                haskey(ss.calibrated, vi) || throw(ArgumentError(
                    "$vi must be a calibrated variable for the steady state"))
            else
                v in srcs(m) || throw(ArgumentError("$vi is not a source of the model"))
            end
            vi in jacus && throw(ArgumentError("inputs cannot contain any Jacobian unknown"))
        end
        for vo in outs
            v = get(m.invpool, vo, 0)
            v === 0 && throw(ArgumentError("$vo is not a variable of the model"))
            # outlength rely on varvals
            haskey(ss[], vo) || error("$vo is not in varvals")
        end
        for v in jactars
            hastarget(ss, v) || throw(ArgumentError("$v is not a target for steady state"))
            v in jacus && throw(ArgumentError("Jacobian unknowns and targets cannot overlap"))
        end
        # If static==true, inputs should not involve temporal terms but this is not verified
        return new{NJ,ST,SS,typeof(sscache),LS,ins,outs}(
            ssins, ss, Ref(sscache), ssargs, jacus, jactars, jacargs, static, sparseH_U)
    end
end

# Allow irrelevant kwargs for @implicit
function block(ss::SteadyState, ins, outs;
        solver=NoRootSolver, ssins=ins, ssargs=nothing,
        jacus=inputs(ss), jactars=targets(ss), jacargs=nothing, static=false,
        sparseH_U=false, linsolvertype=default_linsolvertype(sparseH_U), kwargs...)
    ins = ins isa Union{Symbol,VarSpec} ? (ins,) : (ins...,)
    outs = outs isa Symbol ? (outs,) : (outs...,)
    ssins isa Union{Symbol,VarSpec} && (ssins = (ssins,))
    ins = map(name, ins)
    ssins = Set{Symbol}(map(name, ssins))
    outs = map(name, outs)
    jacus = jacus isa Symbol ? (jacus,) : (jacus...,)
    jactars = jactars isa Symbol ? (jactars,) : (jactars...,)
    ssargs = ssargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(ssargs...)
    jacargs = jacargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(jacargs...)
    return CombinedBlock(ins, ssins, outs, ss, solver, ssargs, jacus, jactars, jacargs,
        static, sparseH_U, linsolvertype)
end

function block(bs::Union{AbstractBlock,Vector{<:AbstractBlock}}, ins, outs,
        calibrated::ValidVarInput, initials::Union{ValidVarInput,Nothing}=nothing,
        tars::Union{ValidVarInput,Nothing}=nothing, TF::Type=Float64;
        jacus=nothing, jactars=nothing, static=false, sparseH_U=false,
        linsolvertype=default_linsolvertype(sparseH_U), kwargs...)
    ss = SteadyState(model(bs), calibrated, initials, tars, TF)
    jacus === nothing && (jacus = inputs(ss))
    jactars === nothing && (jactars = targets(ss))
    return block(ss, ins, outs; jacus=jacus, jactars=jactars, static=static,
        sparseH_U=sparseH_U, linsolvertype=linsolvertype, kwargs...)
end

invars(b::CombinedBlock) = inputs(b)

hasjactars(::CombinedBlock{NJ}) where NJ = NJ > 0
solvertype(::CombinedBlock{NJ,ST}) where {NJ,ST} = ST
linsolvertype(::CombinedBlock{NJ,ST,SS,CA,LS}) where {NJ,ST,SS,CA,LS} = LS

# Use varvals attached to b.ss instead
outlength(b::CombinedBlock, varvals::NamedTuple) =
    sum(vo->length(b.ss[vo]), outputs(b))
outlength(b::CombinedBlock, varvals::NamedTuple, r::Int) =
    length(b.ss[outputs(b)[r]])

model(b::CombinedBlock) = model(b.ss)

function steadystate!(b::CombinedBlock, varvals::NamedTuple)
    b.ss.varvals[] = merge(b.ss[], NamedTuple{inputs(b)}(varvals))
    if b.sscache isa RefValue{Nothing}
        bvarvals, _, found = _solve!(solvertype(b), b.ss; b.ssargs...)
    else
        bvarvals, r, found = _solve!(b.sscache[], b.ss; b.ssargs...)
        b.sscache[] = r
    end
    found || error("failed to solve steady state")
    return merge(varvals, NamedTuple{outputs(b)}(bvarvals))
end

struct CombinedBlockJacobian{BLK<:CombinedBlock, TF, G<:GMaps,
        P<:GEJacobianUpdatePlan} <: MatrixBlockJacobian{TF}
    blk::BLK
    J::Matrix{Matrix{TF}}
    Gs::G
    iins::Vector{Int}
    plan::P
    ishiftmap::Vector{Pair{Int,ShiftMap{TF}}}
end

function (j::CombinedBlockJacobian)(varvals::NamedTuple)
    b = j.blk
    j.blk.ss.varvals[] = invarvals = merge(b.ss[], NamedTuple{inputs(b)}(varvals))
    j.Gs.gj.tjac.varvals[] = invarvals # Not used for computation but just to be safe
    j.plan(invarvals)
    j.Gs()
    for (i, smap) in j.ishiftmap
        mul!(j.J[i], smap, true)
    end
end

function jacobian(b::CombinedBlock, iins, nT::Int, varvals::NamedTuple, TF::Type=Float64)
    ins = inputs(b)
    outs = outputs(b)
    iins = collect(iins)
    exos = map(i->ins[i], iins)
    sources = (exos..., b.jacus...)
    excluded = get(b.jacargs, :excluded, nothing)
    # Set nT = 1 and avoid WrappedMap for static problems
    nTfull = nT
    b.static && (nT = 1)
    # varvals from the argument may not contain internal variabls of b
    b.ss.varvals[] = invarvals = merge(b.ss[], NamedTuple{ins}(varvals))
    j = TotalJacobian(model(b.ss), sources, b.jactars, invarvals, nT; excluded=excluded)
    gj = GEJacobian(j, exos; nTfull=nTfull, sparseH_U=b.sparseH_U,
        linsolvertype=linsolvertype(b))
    Gs = GMaps(gj, outs)
    # Parameters should not have been reached and hence used the unreached ins
    p = plan(gj, setdiff(ins, exos))
    No = length(outs)
    Ni = length(iins)
    ishiftmap = Pair{Int,ShiftMap{TF}}[]
    # Turn jacmap to Matrix in order to get combined with jacmap from outer blocks
    J = Matrix{Matrix{TF}}(undef, No, Ni)
    for n in 1:Ni
        vi = exos[n]
        for m in 1:No
            vo = outs[m]
            map = get(Gs[vi], vo, nothing)
            if map === nothing
                J[m,n] = zeros(TF, nT*length(invarvals[vo]), nT*length(invarvals[vi]))
            elseif map isa ShiftMap
                J[m,n] = map(nT)
                push!(ishiftmap, LinearIndices(J)[m,n]=>map)
            else
                J[m,n] = map.out
            end
        end
    end
    return CombinedBlockJacobian(b, J, Gs, iins, p, ishiftmap)
end

@inline getindex(j::CombinedBlockJacobian, r::Int, i::Int) = j.J[r,i]

struct PECombinedBlockJacobian{BLK<:CombinedBlock, TF, BJ<:Tuple,
        NT} <: MatrixBlockJacobian{TF}
    blk::BLK
    bjs::BJ
    tjac::TotalJacobian{TF,NT}
    smaps::Vector{ShiftMap{TF}}
    mmaps::Vector{Vector{MatrixMap{TF}}}
    J::Matrix{Matrix{TF}}
    iins::Vector{Int}
    ishiftmap::Vector{Pair{Int,ShiftMap{TF}}}
end

@inline function _update_blkjacs!(p::PECombinedBlockJacobian{BLK,TF,BJ},
        varvals::NamedTuple) where {BLK,TF,BJ}
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

function (j::PECombinedBlockJacobian)(varvals::NamedTuple)
    b = j.blk
    j.blk.ss.varvals[] = merge(b.ss[], NamedTuple{inputs(b)}(varvals))
    # Do something like GEJacobianUpdatePlan
    # This may involve updating irrelevant blocks
    _update_blkjacs!(j, varvals)
    for smap in j.smaps
        _updateout!(smap)
    end
    for Mmaps in j.mmaps
        # ins for the other Mmaps should have been handled due to the sharing of arrays
        _updateins!(Mmaps[1])
        for Mmap in Mmaps
            _updateout!(Mmap)
        end
    end
    for (i, smap) in j.ishiftmap
        mul!(j.J[i], smap, true)
    end
end

function jacobian(b::CombinedBlock{0}, iins, nT::Int, varvals::NamedTuple;
        dZs=nothing, TF::Type=Float64)
    ins = inputs(b)
    outs = outputs(b)
    iins = collect(iins)
    exos = map(i->ins[i], iins)
    sources = (exos..., b.jacus...)
    excluded = get(b.jacargs, :excluded, nothing)
    b.ss.varvals[] = invarvals = merge(b.ss[], NamedTuple{ins}(varvals))
    j = TotalJacobian(model(b.ss), sources, b.jactars, invarvals, nT;
        dZs=dZs, excluded=excluded)
    # Do something similar to GEJacobianUpdatePlan but update all blocks
    # without removing blocks that are not affected
    smaps = ShiftMap{TF}[]
    mmaps = Vector{MatrixMap{TF}}[]
    m = j.parent
    bjs = []
    for vi in exos
        d = j.totals[vi]
        for v in m.order
            blk = m.pool[v]
            blk isa Symbol && continue
            bj = j.blkjacs[j.lookupblk[blk]]
            push!(bjs, bj)
            if bj isa ShiftBlockJacobian
                push!(smaps, (d[vo] for vo in outputs(blk) if haskey(d, vo))...)
            else
                mmapaffected = MatrixMap{TF}[d[vo] for vo in outputs(blk) if haskey(d, vo)]
                # ins for variables from the same block are identical
                push!(mmaps, mmapaffected)
            end
        end
    end
    bjs = (unique!(bjs)...,) # Order does not matter?
    No = length(outs)
    Ni = length(iins)
    ishiftmap = Pair{Int,ShiftMap{TF}}[]
    # Turn jacmap to Matrix in order to get combined with jacmap from outer blocks
    J = Matrix{Matrix{TF}}(undef, No, Ni)
    for n in 1:Ni
        vi = exos[n]
        for m in 1:No
            vo = outs[m]
            map = get(j.totals[vi], vo, nothing)
            if map === nothing
                J[m,n] = zeros(TF, nT*length(invarvals[vo]), nT*length(invarvals[vi]))
            elseif map isa ShiftMap
                J[m,n] = map(nT)
                push!(ishiftmap, LinearIndices(J)[m,n]=>map)
            else
                J[m,n] = map.out
            end
        end
    end
    return PECombinedBlockJacobian(b, bjs, j, smaps, mmaps, J, iins, ishiftmap)
end

@inline getindex(j::PECombinedBlockJacobian, r::Int, i::Int) = j.J[r,i]

show(io::IO, b::CombinedBlock{NJ,ST}) where {NJ,ST} =
    (print(io, "CombinedBlock($ST, "); join(io, b.ss.blks, ", "); print(io, ')'))

function show(io::IO, ::MIME"text/plain", b::CombinedBlock{NJ,ST,SS}) where {NJ,ST,SS}
    print(io, "CombinedBlock($ST) with $(b.ss) and $NJ GE restriction")
    println(io, NJ>1 ? "s:" : ":")
    _showinouts(io, b)
    nblk = length(b.ss.blks)
    if nblk > 0
        print(io, "\n  block", nblk>1 ? "s:  " : ":  ")
        join(io, b.ss.blks, ", ")
    end
end

function show(io::IO, b::CombinedBlock{0,ST}) where ST
    print(io, "CombinedBlock($ST, ")
    join(io, (b for b in model(b.ss).pool if b isa AbstractBlock), ", ")
    print(io, ')')
end

function show(io::IO, ::MIME"text/plain", b::CombinedBlock{0,ST,SS}) where {ST,SS}
    println(io, "CombinedBlock($ST) with $(b.ss) and 0 GE restriction:")
    _showinouts(io, b)
    nblk = length(b.ss.blks)
    blks = [b for b in model(b.ss).pool if b isa AbstractBlock]
    nblk = length(blks)
    if nblk > 0
        print(io, "\n  block", nblk>1 ? "s:  " : ":  ")
        join(io, blks, ", ")
    end
end

function show(io::IO, j::CombinedBlockJacobian)
    print(io, "CombinedBlockJacobian(")
    join(io, j.plan.gj.tjac.tars, ", ")
    print(io, ": ")
    _show_jac_from_to(io, j)
    print(io, ')')
end

function show(io::IO, j::PECombinedBlockJacobian)
    print(io, "PECombinedBlockJacobian(")
    _show_jac_from_to(io, j)
    print(io, ')')
end
