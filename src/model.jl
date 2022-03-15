const ValidVarInput = Union{Pair{Symbol,<:ValType}, Vector{<:Pair{Symbol,<:ValType}},
    Dict{Symbol,<:ValType}}
const BlockOrVar = Union{AbstractBlock, Symbol}

struct SequenceSpaceModel <: AbstractGraph{Int}
    dag::SimpleDiGraph{Int}
    pool::Vector{BlockOrVar}
    invpool::Dict{BlockOrVar,Int}
    order::Vector{Int}
    srcs::Set{Int}
    sssrcs::Set{Int}
    dests::Set{Int}
    function SequenceSpaceModel(dag::SimpleDiGraph{Int}, pool::Vector{BlockOrVar},
            invpool::Dict{BlockOrVar,Int})
        order = topological_sort_by_dfs(dag)
        srcs = Set{Int}()
        sssrcs = Set{Int}()
        dests = Set{Int}()
        for v in vertices(dag)
            pool[v] isa AbstractBlock && continue
            ins = inneighbors(dag, v)
            if isempty(ins)
                push!(srcs, v)
                for o in outneighbors(dag, v)
                    if pool[v] in ssinputs(pool[o])
                        push!(sssrcs, v)
                        break
                    end
                end
            elseif length(ins) > 1
                var = pool[v]
                error("$var is the output of $(length(ins)) blocks")
            elseif isempty(outneighbors(dag, v))
                push!(dests, v)
            end
        end
        return new(dag, pool, invpool, order, srcs, sssrcs, dests)
    end
end

function model(blocks)
    blocks isa BlockOrVar && (blocks = (blocks,))
    pool = collect(BlockOrVar, blocks)
    n = length(pool)
    invpool = Dict{BlockOrVar,Int}(pool.=>1:n)
    edges = Edge{Int}[]
    for (ib, b) in enumerate(blocks)
        vins = inputs(b)
        vouts = outputs(b)
        nvin = length(vins)
        for (c, v) in enumerate((vins..., vouts...))
            iv = get(invpool, v, 0)
            if iv === 0
                n += 1
                push!(pool, v)
                invpool[v] = n
                iv = n
            end
            push!(edges, c<=nvin ? Edge(iv,ib) : Edge(ib,iv))
        end
    end
    dag = SimpleDiGraphFromIterator(edges)
    return SequenceSpaceModel(dag, pool, invpool)
end

SimpleDiGraph(m::SequenceSpaceModel) = m.dag

edgetype(m::SequenceSpaceModel) = edgetype(m.dag)
eltype(m::SequenceSpaceModel) = eltype(m.dag)
nv(m::SequenceSpaceModel) = nv(m.dag)
ne(m::SequenceSpaceModel) = ne(m.dag)
vertices(m::SequenceSpaceModel) = vertices(m.dag)
edges(m::SequenceSpaceModel) = edges(m.dag)
is_directed(::Type{SequenceSpaceModel}) = true
has_vertex(m::SequenceSpaceModel, v) = has_vertex(m.dag, v)
has_edge(m::SequenceSpaceModel, s, d) = has_edge(m.dag, s, d)
inneighbors(m::SequenceSpaceModel, v) = inneighbors(m.dag, v)
outneighbors(m::SequenceSpaceModel, v) = outneighbors(m.dag, v)
zero(::Type{SequenceSpaceModel}) =
    SequenceSpaceModel(zero(SimpleDiGraph), BlockOrVar[], Dict{BlockOrVar,Int}())

srcs(m::SequenceSpaceModel) = getfield(m, :srcs)
sssrcs(m::SequenceSpaceModel) = getfield(m, :sssrcs)
dests(m::SequenceSpaceModel) = getfield(m, :dests)
isblock(m::SequenceSpaceModel, v) = m.pool[v] isa AbstractBlock

show(io::IO, m::SequenceSpaceModel) = print(io, "{$(nv(m)), $(ne(m))} ", typeof(m).name.name)

function show(io::IO, ::MIME"text/plain", m::SequenceSpaceModel)
    isb = isa.(m.pool, AbstractBlock)
    nb = sum(isb)
    nvar = length(m.pool) - nb
    print(io, "{$(nv(m)), $(ne(m))} ", typeof(m).name.name)
    print(io, " with $nb block")
    nb > 1 && print(io, 's')
    print(io, " and $nvar variable")
    nvar > 1 && print(io, 's')
    if nb > 0
        print(io, ':')
        for b in view(m.pool, isb)
            print(io, "\n  ", b)
        end
    end
end

struct SteadyState{TF<:AbstractFloat}
    parent::SequenceSpaceModel
    blks::Vector{AbstractBlock}
    vars::Vector{Symbol}
    ins::Vector{Symbol}
    outs::Vector{Symbol}
    calis::Dict{Symbol,ValType{TF}}
    tars::Dict{Symbol,ValType{TF}}
    varvals::Dict{Symbol,ValType{TF}}
    inits::Vector{TF}
    resids::Vector{TF}
    targeted::BitVector
end

function _collapse!(m::SequenceSpaceModel, calis::Dict, varvals::Dict)
    unknowns = Set{Int}()
    ins = Symbol[]
    outs = Symbol[]
    blks = AbstractBlock[]
    visitedsrc = 0
    Nsrc = length(srcs(m))
    for v in m.order
        if isblock(m, v)
            b = m.pool[v]
            gins = inneighbors(m.dag, v)
            for gi in gins
                if gi in unknowns
                    push!(unknowns, v)
                    push!(blks, b)
                    break
                end
            end
            steadystate!(varvals, b)
        else
            var = m.pool[v]
            cali = get(calis, var, nothing)
            if visitedsrc < Nsrc && v in srcs(m)
                visitedsrc += 1
                if v in sssrcs(m) && cali === nothing
                    msg = "initial value for input variable $var is not assigned"
                    haskey(varvals, var) || @warn msg
                    push!(unknowns, v)
                    push!(ins, var)
                end
            else
                gin = inneighbors(m.dag, v)[1]
                if gin in unknowns
                    push!(unknowns, v)
                    v in dests(m) && push!(outs, var)
                    if cali !== nothing
                        @warn "calibrated value for $var is replaced by output from its parent block"
                        delete!(calis, var)
                    end
                end
            end
            # Only assign calibrated values after the checks
            cali === nothing || (varvals[var] = cali)
        end
    end
    return ins, outs, blks
end

function SteadyState(m::SequenceSpaceModel, calibrated::ValidVarInput,
        targets::ValidVarInput, initials::Union{ValidVarInput,Nothing}=nothing,
        TF::Type=Float64)
    calibrated isa Pair && (calibrated = (calibrated,))
    targets isa Pair && (targets = (targets,))
    initials !== nothing && initials isa Pair && (initials = (initials,))
    calis = Dict{Symbol,ValType{TF}}(calibrated...)
    vars = Symbol[v for v in m.pool if v isa Symbol]
    varvals = Dict{Symbol,ValType{TF}}()
    # Assign initials to varvals helps look up assigned values when filling inits
    # This is also the way to tell whether any unknown variable is an Array
    if initials !== nothing
        for (k, v) in initials
            varvals[k] = v
        end
    end

    ins, outs, blks = _collapse!(m, calis, varvals)

    tars = Dict{Symbol,ValType{TF}}()
    for (k, v) in targets
        if k in outs
            tars[k] = v
        else
            @warn "target value for $k is ignored because $k is an input of a block; consider adding a block for the gap between the target"
        end
    end
    # Assign a scalar zero whenever the initial value is not provided
    z = zero(TF)
    inits = Vector{TF}(undef, length(ins))
    i0 = 0
    if initials !== nothing
        @inbounds for vi in ins
            val = get(varvals, vi, z)
            w = length(val)
            if w > 1
                resize!(inits, length(inits)+w-1)
                inits[i0+1:i0+w] .= val 
            else
                inits[i0+1] = val
            end
            i0 += w
        end
    end
    # Some variables in outs might not be targeted
    resids = Vector{TF}(undef, length(tars))
    targeted = haskey.(Ref(tars), outs)
    return SteadyState(m, blks, vars, ins, outs, calis, tars, varvals, inits, resids, targeted)
end

function _inputs!(ss::SteadyState, inputs::AbstractVector)
    N = length(inputs)
    i0 = 0
    @inbounds for vi in ss.ins
        val = get(ss.varvals, vi, nothing)
        w = length(val)
        i0+w > N && error("length of inputs does not match varvals")
        if val isa Vector
            for i in 1:w
                val[i] = inputs[i0+i]
            end
        else
            ss.varvals[vi] = inputs[i0+1]
        end
        i0 += w
    end
end

function _resids!(resids::AbstractVector, ss::SteadyState)
    i0 = 0
    @inbounds for out in ss.outs
        tar = get(ss.tars, out, nothing)
        tar === nothing && continue
        val = ss.varvals[out]
        w = length(val)
        ilast = i0 + w
        ilast > length(resids) && resize!(resids, ilast)
        if val isa Vector
            # tar might be a scalar
            resids[i0+1:i0+w] .= val .- tar
        else
            resids[i0+1] = val - tar
        end
        i0 += w
    end
end

function residuals!(resids::AbstractVector, ss::SteadyState)
    for b in ss.blks
        steadystate!(ss.varvals, b)
    end
    _resids!(resids, ss)
    return resids
end

residuals!(ss::SteadyState) = residuals!(ss.resids, ss)

function residuals!(resids::AbstractVector, ss::SteadyState, inputs::AbstractVector)
    _inputs!(ss, inputs)
    return residuals!(resids, ss)
end

residuals!(ss::SteadyState, inputs::AbstractVector) =
    residuals!(ss.resids, ss, inputs)

function criterion!(ss::SteadyState; weight::Union{AbstractMatrix,UniformScaling}=I)
    resids = residuals!(ss)
    return resids'*weight*resids
end

function criterion!(ss::SteadyState, inputs::AbstractVector;
        weight::Union{AbstractMatrix,UniformScaling}=I)
    resids = residuals!(ss, inputs)
    return resids'*weight*resids
end
