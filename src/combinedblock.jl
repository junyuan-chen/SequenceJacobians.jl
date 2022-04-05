struct CombinedBlock{HasJacTars, ST<:AbstractRootSolver, TF<:AbstractFloat} <: AbstractBlock
    ins::Vector{Symbol}
    ssins::Set{Symbol}
    outs::Vector{Symbol}
    model::SequenceSpaceModel
    ss::SteadyState{TF}
    ssargs::Dict{Symbol,Any}
    jactars::Vector{Symbol}
    jacargs::Dict{Symbol,Any}
    function CombinedBlock{HasJacTars,ST,TF}(ins::Vector{Symbol}, ssins::Set{Symbol},
            outs::Vector{Symbol}, model::SequenceSpaceModel, ss::SteadyState{TF},
            ssargs::Dict{Symbol,Any}, jactars::Vector{Symbol},
            jacargs::Dict{Symbol,Any}) where {HasJacTars,ST,TF}
        ss.parent === model || throw(ArgumentError(
            "the SteadyState is not associated with the specifided model"))
        if ST === NoRootSolver
            isempty(ss.tars) || throw(ArgumentError(
                "NoRootSolver is not allowed with nonempty steady state targets"))
            HasJacTars && throw(ArgumentError(
                "NoRootSolver is not allowed with nonempty Jacobian targets"))
        else
            isempty(ss.tars) && throw(ArgumentError(
                "solver type must be NoRootSolver with empty steady state target"))
        end
        length(ins) > 0 || throw(ArgumentError("the inputs of a block cannot be empty"))
        length(outs) > 0 || throw(ArgumentError("the outputs of a block cannot be empty"))
        for vi in ins
            v = get(model.invpool, vi, 0)
            v === 0 && throw(ArgumentError("$vi is not a variable of the model"))
            if isempty(ss.tars)
                v in srcs(model) || throw(ArgumentError("$vi is not a source of the model"))
            else
                haskey(ss.calis, vi) || throw(ArgumentError(
                    "$vi must be a calibrated variable for the steady state"))
            end
        end
        for vo in outs
            v = get(model.invpool, vo, 0)
            v === 0 && throw(ArgumentError("$vo is not a variable of the model"))
            # nouts rely on varvals
            haskey(ss.varvals, vo) || error("varvals is not populated")
        end
        for v in jactars
            haskey(ss.tars, v) || throw(ArgumentError("$v is not a target for steady state"))
        end
        return new{HasJacTars,ST,TF}(ins, ssins, outs, model, ss, ssargs, jactars, jacargs)
    end
end

function block(m::SequenceSpaceModel, ss::SteadyState{TF}, ins, outs, jactars;
        ST::Type{<:AbstractRootSolver}=NoRootSolver, ssins=ins, ssargs=nothing,
        jacargs=nothing) where TF
    ins isa Union{Symbol,VarSpec} && (ins = (ins,))
    outs isa Symbol && (outs = (outs,))
    ssins isa Union{Symbol,VarSpec} && (ssins = (ssins,))
    ins = name.(ins)
    ssins = Set{Symbol}(name.(ssins))
    outs = collect(Symbol, outs)
    jactars isa Symbol && (jactars = (jactars,))
    jactars = collect(Symbol, jactars)
    HasJacTars = !isempty(jactars)
    ssargs = ssargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(ssargs...)
    jacargs = jacargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(jacargs...)
    return CombinedBlock{HasJacTars,ST,TF}(ins, ssins, outs, m, ss, ssargs, jactars, jacargs)
end

function block(bs::Union{AbstractBlock,Vector{<:AbstractBlock}}, ins, outs, jactars,
        calibrated::ValidVarInput, targets::Union{ValidVarInput,Nothing}=nothing,
        initials::Union{ValidVarInput,Nothing}=nothing, TF::Type=Float64; kwargs...)
    m = model(bs)
    ss = SteadyState(m, calibrated, targets, initials, TF)
    return block(m, ss, ins, outs, jactars; kwargs...)
end

invars(b::CombinedBlock) = inputs(b)

hasjactars(::CombinedBlock{HasJacTars}) where HasJacTars = HasJacTars
solvertype(::CombinedBlock{HasJacTars,ST}) where {HasJacTars,ST} = ST

nouts(b::CombinedBlock) = sum(vo->length(b.ss.varvals[vo]), outputs(b))
outlength(b::CombinedBlock, r::Int) = length(b.ss.varvals[outputs(b)[r]])

function steadystate!(b::CombinedBlock, varvals::AbstractDict)
    bvarvals = b.ss.varvals
    for vi in ssinputs(b)
        bvarvals[vi] = varvals[vi]
    end
    solve!(solvertype(b), b.ss; b.ssargs...)
    for vo in outputs(b)
        val = get(varvals, vo, nothing)
        val isa AbstractArray ? copyto!(val, bvarvals[vo]) : (varvals[vo] = bvarvals[vo])
    end
end

jacbyinput(::CombinedBlock) = false

function jacobian(b::CombinedBlock{true}, nT::Int, varvals::Dict{Symbol})
    ins = inputs(b)
    outs = outputs(b)
    sources = union(ins, outs)
    excluded = get(b.jacargs, :excluded, nothing)
    tjac = TotalJacobian(b.model, sources, b.jactars, b.ss.varvals, nT; excluded=excluded)
    keepH_U = get(b.jacargs, :keepH_U, false)
    keepfactor = get(b.jacargs, :keepfactor, false)
    gejac = GEJacobian(tjac, ins; keepH_U=keepH_U, keepfactor=keepfactor)
    for vi in ins
        for vo in outs
            getG!(gejac, vi, vo)
        end
    end
    return gejac
end

function jacobian(b::CombinedBlock{false}, nT::Int, varvals::Dict{Symbol})
    ins = inputs(b)
    outs = outputs(b)
    sources = union(ins, outs)
    excluded = get(b.jacargs, :excluded, nothing)
    return TotalJacobian(b.model, sources, b.jactars, b.ss.varvals, nT; excluded=excluded)
end

function getjacmap(b::CombinedBlock{true}, J::GEJacobian,
        i::Int, ii::Int, r::Int, rr::Int, nT::Int)
    vi = inputs(b)[i]
    vo = outputs(b)[r]
    return LinearMap(J.Gs[vi][vo]), false
end

function getjacmap(b::CombinedBlock{false}, J::TotalJacobian,
        i::Int, ii::Int, r::Int, rr::Int, nT::Int)
    vi = inputs(b)[i]
    vo = outputs(b)[r]
    return J.totals[vi][vo], false
end
