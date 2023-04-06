rootsolvercache(::Any, ::SteadyState; kwargs...) = nothing

struct CombinedBlock{NJ, ST, SS<:SteadyState, CA, ins, outs} <: AbstractBlock{ins,outs}
    ssins::Set{Symbol}
    ss::SS
    sscache::RefValue{CA}
    ssargs::Dict{Symbol,Any}
    jacus::NTuple{NJ,Symbol}
    jactars::NTuple{NJ,Symbol}
    jacargs::Dict{Symbol,Any}
    static::Bool
    function CombinedBlock(ins::NTuple{NI,Symbol}, ssins::Set{Symbol},
            outs::NTuple{NO,Symbol}, ss::SS, solver, ssargs::Dict{Symbol,Any},
            jacus::NTuple{NJ,Symbol}, jactars::NTuple{NJ,Symbol},
            jacargs::Dict{Symbol,Any}, static::Bool) where {NI,NO,SS<:SteadyState,NJ}
        if isrootsolver(solver)
            ST = solver isa Type ? solver : typeof(solver)
            sscache = rootsolvercache(ST, ss; ssargs...)
        elseif isrootsolvercache(solver)
            ST = typeof(solver)
            sscache = solver
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
        return new{NJ,ST,SS,typeof(sscache),ins,outs}(
            ssins, ss, Ref(sscache), ssargs, jacus, jactars, jacargs, static)
    end
end

# Allow irrelevant kwargs for @implicit
function block(ss::SteadyState, ins, outs;
        solver=NoRootSolver, ssins=ins, ssargs=nothing,
        jacus=inputs(ss), jactars=targets(ss), jacargs=nothing, static=false, kwargs...)
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
        static)
end

function block(bs::Union{AbstractBlock,Vector{<:AbstractBlock}}, ins, outs,
        calibrated::ValidVarInput, initials::Union{ValidVarInput,Nothing}=nothing,
        tars::Union{ValidVarInput,Nothing}=nothing, TF::Type=Float64;
        jacus=nothing, jactars=nothing, static=false, kwargs...)
    ss = SteadyState(model(bs), calibrated, initials, tars, TF)
    jacus === nothing && (jacus = inputs(ss))
    jactars === nothing && (jactars = targets(ss))
    return block(ss, ins, outs; jacus=jacus, jactars=jactars, static=static, kwargs...)
end

invars(b::CombinedBlock) = inputs(b)

hasjactars(::CombinedBlock{NJ}) where NJ = NJ > 0
solvertype(::CombinedBlock{NJ,ST}) where {NJ,ST} = ST

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
        P<:GEJacobianUpdatePlan} <: AbstractBlockJacobian{TF}
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
    j = TotalJacobian(model(b.ss), sources, b.jactars, invarvals, nT;
        excluded=excluded)
    gj = GEJacobian(j, exos; nTfull=nTfull)
    Gs = GMaps(gj, outs)
    p = plan(gj, exos)
    No = length(outs)
    Ni = length(iins)
    ishiftmap = Pair{Int,ShiftMap{TF}}[]
    J = Matrix{Matrix{TF}}(undef, No, Ni)
    for n in 1:Ni
        vi = exos[n]
        for m in 1:No
            vo = outs[m]
            map = get(Gs[vi], vo, nothing)
            if map === nothing
                J[m,n] = Zeros{TF}(nT*length(invarvals[vo]), nT*length(invarvals[vi]))
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

# ! To do: Support in-place update via CombinedBlockJacobian?
#=
function jacobian(b::CombinedBlock{0}, iins::NTuple{N,Int}, nT::Int,
        varvals::NamedTuple) where N
    ins = map(i->inputs(b)[i], iins)
    excluded = get(b.jacargs, :excluded, nothing)
    b.ss.varvals[] = varvals = merge(b.ss[], NamedTuple{ins}(varvals))
    j = TotalJacobian(model(b.ss), ins, b.jactars, varvals, nT; excluded=excluded)
    return CombinedBlockJacobian(b, j, iins)
end
=#

@inline getindex(j::CombinedBlockJacobian, r::Int, i::Int) = j.J[r,i]

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

function show(io::IO, j::CombinedBlockJacobian)
    print(io, "CombinedBlockJacobian(")
    join(io, j.plan.gj.tjac.tars, ", ")
    print(io, ": ")
    _show_jac_from_to(io, j)
    print(io, ')')
end
