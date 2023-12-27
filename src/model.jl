const ValidVarInput = Union{Pair{Symbol,<:Any}, Vector{<:Pair{Symbol,<:Any}},
    Dict{Symbol,<:Any}}
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
vsrcs(m::SequenceSpaceModel) = map(i->m.pool[i], sort!(collect(srcs(m))))
sssrcs(m::SequenceSpaceModel) = getfield(m, :sssrcs)
vsssrcs(m::SequenceSpaceModel) = map(i->m.pool[i], sort!(collect(sssrcs(m))))
dests(m::SequenceSpaceModel) = getfield(m, :dests)
vdests(m::SequenceSpaceModel) = map(i->m.pool[i], sort!(collect(dests(m))))
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

struct SteadyState{TF<:AbstractFloat, NT<:NamedTuple, BLK<:Tuple, scins, arins, sctars, artars}
    parent::SequenceSpaceModel
    blks::BLK
    vars::Vector{Symbol}
    varvals::RefValue{NT}
    inits::Vector{TF}
    arinrange::Vector{UnitRange{Int}}
    tars::Vector{TF}
    artarrange::Vector{UnitRange{Int}}
    resids::Vector{TF}
    outs::Vector{Symbol}
    targeted::BitVector
    calibrated::Dict{Symbol,Any}
    targets::Dict{Symbol,Any}
end

function _collapse!(m::SequenceSpaceModel, calis::Dict, tars::Dict, varvals::NamedTuple)
    unknowns = Set{Int}()
    scins = Symbol[]
    arins = Symbol[]
    outs = Symbol[]
    sctars = Symbol[]
    artars = Symbol[]
    blks = []
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
            varvals = steadystate!(b, varvals)
        else
            var = m.pool[v]
            cali = get(calis, var, nothing)
            if visitedsrc < Nsrc && v in srcs(m)
                visitedsrc += 1
                if v in sssrcs(m) && cali === nothing
                    haskey(varvals, var) ||
                        error("initial value for input variable $var is not assigned")
                    push!(unknowns, v)
                    val = varvals[var]
                    push!(val isa Real ? scins : arins, var)
                    if haskey(tars, var)
                        @warn "target $var is ignored as it is a source"
                        delete!(tars, var)
                    end
                end
            else
                gin = inneighbors(m.dag, v)[1]
                if gin in unknowns
                    push!(unknowns, v)
                    v in dests(m) && push!(outs, var)
                    if haskey(tars, var)
                        val = varvals[var]
                        length(val) == length(tars[var]) || throw(DimensionMismatch(
                            "length of target $var is $(length(val)); got $(length(tars[var]))"))
                        push!(val isa Real ? sctars : artars, var) 
                    end
                    if cali !== nothing
                        @warn "calibrated value for $var is replaced by output from its parent block"
                        delete!(calis, var)
                    end
                end
            end
            # Only assign calibrated values after the checks
            cali === nothing || (varvals = merge(varvals, (; var=>cali)))
        end
    end
    return scins, arins, sctars, artars, outs, blks, varvals
end

function SteadyState(m::SequenceSpaceModel, calibrated::ValidVarInput,
        initials::Union{ValidVarInput,Nothing}=nothing,
        targets::Union{ValidVarInput,Nothing}=nothing, TF::Type=Float64)
    calibrated isa Pair && (calibrated = (calibrated,))
    initials !== nothing && initials isa Pair && (initials = (initials,))
    initials === nothing && (initials = ())
    targets !== nothing && targets isa Pair && (targets = (targets,))
    targets === nothing && (targets = ())
    # The orders of names in inputs are not preserved
    calibrated = Dict{Symbol,Any}(calibrated...)
    targets = Dict{Symbol,Any}(targets...)
    vars = Symbol[v for v in m.pool if v isa Symbol]
    varvals = NamedTuple()
    # Assign initials to varvals helps look up assigned values when filling inits
    # This is also the way to tell whether any unknown variable is an Array
    if !isempty(initials)
        for (k, v) in initials
            if v isa Real
                v = convert(TF, v)
            elseif v isa AbstractArray
                v = convert(AbstractArray{TF}, v)
            end
            varvals = merge(varvals, (; k=>v))
        end
    end

    scins, arins, sctars, artars, outs, blks, varvals =
        _collapse!(m, calibrated, targets, varvals)

    inlength = length(scins)
    isempty(arins) || (inlength += sum(x->length(varvals[x]), arins))

    inits = Vector{TF}(undef, inlength)
    arinrange = Vector{UnitRange{Int}}(undef, length(arins))
    if !isempty(initials)
        i0 = 0
        @inbounds for vi in scins
            i0 += 1
            val = varvals[vi]
            inits[i0] = val
        end
        for (iarin, vi) in enumerate(arins)
            val = varvals[vi]
            w = length(val)
            r = i0+1:i0+w
            copyto!(view(inits, r), view(val, :))
            arinrange[iarin] = r
            i0 += w
        end
    end

    tarlength = length(sctars)
    isempty(artars) || (tarlength += sum(x->length(targets[x]), artars))
    tarlength == inlength || @warn "$inlength inputs for $tarlength targets"

    tars = Vector{TF}(undef, tarlength)
    artarrange = Vector{UnitRange{Int}}(undef, length(artars))
    if !isempty(targets)
        i0 = 0
        @inbounds for vt in sctars
            i0 += 1
            val = targets[vt]
            tars[i0] = val
        end
        for (iartar, vt) in enumerate(artars)
            val = targets[vt]
            w = length(val)
            r = i0+1:i0+w
            copyto!(view(tars, r), view(val, :))
            artarrange[iartar] = r
            i0 += w
        end
    end

    # Some variables in outs might not be targeted
    resids = Vector{TF}(undef, length(tars))
    targeted = haskey.(Ref(targets), outs)
    blks = (blks...,)
    scins = (scins...,)
    arins = (arins...,)
    sctars = (sctars...,)
    artars = (artars...,)
    return SteadyState{TF,typeof(varvals),typeof(blks),scins,arins,sctars,artars}(
        m, blks, vars, Ref(varvals), inits, arinrange, tars, artarrange,
        resids, outs, targeted, calibrated, targets)
end

varvalstype(::SteadyState{TF,NT}) where {TF,NT} = NT
blkstype(::SteadyState{TF,NT,BLK}) where {TF,NT,BLK} = BLK
scalarinputs(::SteadyState{TF,NT,BLK,scins}) where {TF,NT,BLK,scins} = scins
arrayinputs(::SteadyState{TF,NT,BLK,scins,arins}) where {TF,NT,BLK,scins,arins} = arins
scalartargets(::SteadyState{TF,NT,BLK,scins,arins,sctars}) where
    {TF,NT,BLK,scins,arins,sctars} = sctars
arraytargets(::SteadyState{TF,NT,BLK,scins,arins,sctars,artars}) where
    {TF,NT,BLK,scins,arins,sctars,artars} = artars

model(ss::SteadyState) = ss.parent
inputs(ss::SteadyState) = (scalarinputs(ss)..., arrayinputs(ss)...)
inlength(ss::SteadyState) = length(ss.inits)
targets(ss::SteadyState) = (scalartargets(ss)..., arraytargets(ss)...)
tarlength(ss::SteadyState) = length(ss.tars)
hastarget(ss::SteadyState) = tarlength(ss) > 0
hastarget(ss::SteadyState, v::Symbol) = haskey(ss.targets, v)

getindex(ss::SteadyState) = ss.varvals[]
getindex(ss::SteadyState, i) = getindex(ss[], i)

function _inputs!(varvals::NT, ss::SteadyState{TF,NT}, inputs::AbstractVector) where {TF,NT}
    scins = scalarinputs(ss)
    arins = arrayinputs(ss)
    Nsc = length(scins)
    Nar = length(arins)
    scalars = NamedTuple{scins}((view(inputs, 1:Nsc)...,))
    if Nar > 0
        arrays = NamedTuple{arins}(varvals)
        rs = ss.arinrange
        foreach(i->copyto!(view(arrays[i], :), view(inputs, rs[i])), 1:Nar)
    end
    return merge(varvals, scalars)
end

function _tars!(resids::AbstractVector, src::AbstractVector, tars::AbstractVector)
    @simd for i in eachindex(resids)
        @inbounds resids[i] = src[i] - tars[i]
    end
end

function _resids!(resids::AbstractVector, ss::SteadyState{TF,NT}, varvals::NT) where {TF,NT}
    sctars = scalartargets(ss)
    artars = arraytargets(ss)
    Nsc = length(sctars)
    Nar = length(artars)
    scalars = NamedTuple{sctars}(varvals)
    foreach(i->@inbounds(resids[i]=scalars[i]-ss.tars[i]), 1:Nsc)
    if Nar > 0
        arrays = NamedTuple{artars}(varvals)
        rs = ss.artarrange
        foreach(i->_tars!(view(resids,rs[i]), view(arrays[i],:), view(ss.tars,rs[i])), 1:Nar)
    end
    return resids
end

function residuals!(resids::AbstractVector, ss::SteadyState{TF,NT,BLK}, inputs::AbstractVector) where {TF,NT,BLK}
    if @generated
        ex = :(_inputs!(ss[], ss, inputs))
        for i in 1:length(BLK.parameters)
            ex = :(steadystate!(blks[$i], $ex))
        end
        ex = quote
            blks = ss.blks
            varvals = $ex
            ss.varvals[] = varvals
            _resids!(resids, ss, varvals)
        end
        return ex
    else
        varvals = _inputs!(ss[], ss, inputs)
        for b in ss.blks
            varvals = steadystate!(b, varvals)
        end
        ss.varvals[] = varvals
        return _resids!(resids, ss, varvals)
    end
end

residuals!(ss::SteadyState) = residuals!(ss.resids, ss, ss.inits)
residuals!(ss::SteadyState, inputs::AbstractVector) = residuals!(ss.resids, ss, inputs)

(ss::SteadyState)(resids::AbstractVector, inputs::AbstractVector) =
    residuals!(resids, ss, inputs)

# When there is only one equation involved
(ss::SteadyState)(input::Number) =
    (ss.inits[1] = input; residuals!(ss); ss.resids[1])

# Should only be used internally (eg., by CombinedBlock)
function _solve!(ST::Type, ss::SteadyState, x0=ss.inits; keepinits::Bool=false, kwargs...)
    r = isrootsolver(ST) ? solve(ST, ss, x0; kwargs...) : solve!(ST, x0; kwargs...)
    # Results returned by the solver may be the guess for the next iteration
    keepinits || copyto!(ss.inits, root(r))
    return ss[], r, rootisfound(r)
end

#! To do: Model without equilibrium conditions?
#=
function _solve!(ST::Type{NoRootSolver}, ss::SteadyState{TF,NT,BLK};
        kwargs...) where {TF,NT,BLK}
    # Update the values without solving for any target
    if @generated
        ex = :(ss[])
        for i in 1:length(BLK.parameters)
            ex = :(steadystate!(blks[$i], $ex))
        end
        return :(blks = ss.blks; varvals = $ex; ss.varvals[] = varvals; varvals)
    else
        varvals = ss[]
        for b in ss.blks
            varvals = steadystate!(b, varvals)
        end
        ss.varvals[] = varvals
        return varvals
    end
end
=#

show(io::IO, ss::SteadyState{TF}) where TF =
    print(io, inlength(ss), '×', tarlength(ss), " SteadyState{$TF}")

function show(io::IO, ::MIME"text/plain", ss::SteadyState{TF}) where TF
    print(io, inlength(ss), '×', tarlength(ss))
    nblk = length(ss.blks)
    nvar = length(ss.vars)
    print(io, " SteadyState{$TF} with $nblk block")
    nblk > 1 && print(io, 's')
    print(io, " and $nvar variable")
    nvar > 1 && print(io, 's')
    println(io, ":")
    print(io, "  unknowns: ")
    join(io, inputs(ss), ", ")
    print(io, "\n  targets:  ")
    join(io, targets(ss), ", ")
end
