const ValidCache = Union{Dict{Symbol},Nothing}

struct VarSpec
    name::Symbol
    shift::Int
end

var(name::Symbol; shift::Int=0) = VarSpec(name, shift)
var(v::VarSpec) = v

lag(name::Symbol, n=1) = var(name, shift=-n)
lead(name::Symbol, n=1) = var(name, shift=n)

name(v::VarSpec) = getfield(v, :name)
shift(v::VarSpec) = getfield(v, :shift)

name(v::Symbol) = v

convert(::Type{Symbol}, v::VarSpec) = v.name

show(io::IO, v::VarSpec) =
    (s = v.shift; n = v.name; print(io, s>0 ? "lead($n)" : s<0 ? "lag($n)" : n))

abstract type AbstractBlock{ins,outs} end

inputs(::AbstractBlock{ins}) where ins = ins
invars(b::AbstractBlock) = getfield(b, :invars)
ssinputs(b::AbstractBlock) = getfield(b, :ssins)
outputs(::AbstractBlock{ins,outs}) where {ins,outs} = outs

_countcache(cache::ValidCache, outs) =
    sum(k->haskey(cache, k) ? length(cache[k])::Int : 1, outs)

hascache(b::AbstractBlock) = hasfield(typeof(b), :cache)
outlength(b::AbstractBlock) =
    hascache(b) ? _countcache(b.cache, outputs(b)) : length(outputs(b))

function outlength(b::AbstractBlock, r::Int)
    hascache(b) || return 1
    vo = outputs(b)[r]
    return haskey(b.cache, vo) ? length(b.cache[vo])::Int : 1
end

function _checkinsouts(ins, outs, ssins)
    length(ins) > 0 || throw(ArgumentError("the inputs of a block cannot be empty"))
    length(outs) > 0 || throw(ArgumentError("the outputs of a block cannot be empty"))
    ssins ⊆ ins || throw(ArgumentError("ssins must be a subset of the block inputs"))
    isempty(intersect(ins, outs)) ||
        throw(ArgumentError("an input cannot be the output of the same block"))
end

struct SimpleBlock{CA<:ValidCache,F<:Function,ins,outs,NI} <: AbstractBlock{ins,outs}
    invars::NTuple{NI,VarSpec}
    ssins::Set{Symbol}
    cache::CA
    f::F
    function SimpleBlock(ins::NTuple{NI,Symbol}, invars::NTuple{NI,VarSpec},
            ssins::Set{Symbol}, outs::NTuple{NO,Symbol}, cache::CA, f::F) where {NI,NO,CA,F}
        _checkinsouts(ins, outs, ssins)
        return new{CA,F,ins,outs,NI}(invars, ssins, cache, f)
    end
end

function _inout(ins, outs, ssins)
    ins = ins isa Union{Symbol,VarSpec} ? (ins,) : (ins...,)
    outs = outs isa Symbol ? (outs,) : (outs...,)
    ssins isa Union{Symbol,VarSpec} && (ssins = (ssins,))
    # Must do invars before ins
    invars = map(var, ins)
    ins = map(name, ins)
    ssins = Set{Symbol}(map(name, ssins))
    outs = map(name, outs)
    return ins, invars, ssins, outs
end

# Allow irrelevant kwargs for @implicit
block(f::Function, ins, outs; ssins=ins, cache=nothing, kwargs...) =
    SimpleBlock(_inout(ins, outs, ssins)..., cache, f)

hascache(b::SimpleBlock{Nothing}) = false
outlength(b::SimpleBlock{Nothing}) = length(outputs(b))
outlength(b::SimpleBlock{Nothing}, r::Int) = 1

function (b::SimpleBlock)(x...)
    out = hascache(b) ? b.f(b.cache, x...) : b.f(x...)
    out = out isa Tuple ? out : (out,)
    return NamedTuple{outputs(b)}(out)
end

function steadystate!(b::SimpleBlock, varvals::NamedTuple)
    vals = b(map(k->getfield(varvals, k), inputs(b))...)
    return merge(varvals, vals)
end

jacbyinput(::SimpleBlock) = true

function jacobian(b::SimpleBlock, ::Val{i}, nT::Int, varvals::NamedTuple,
        TF::Type=Float64) where i
    ins = inputs(b)
    vi = ins[i]
    val = varvals[vi]
    f(x) = collect(Iterators.flatten(b.f(map(k->getfield(varvals, k), ins[1:i-1])..., x,
        map(k->getfield(varvals, k), ins[i+1:length(ins)])...)))
    no = outlength(b)
    J = Matrix{TF}(undef, no, length(val))
    if val isa Real
        return ForwardDiff.derivative!(J, f, val)
    else
        return ForwardDiff.jacobian!(J, f, vec(val))
    end
end

function getjacmap(b::SimpleBlock{Nothing}, J::Matrix,
        i::Int, ii::Int, r::Int, rr::Int, nT::Int)
    j = J[rr,ii]
    return ShiftMap(Shift(shift(invars(b)[i]), j), nT), iszero(j)
end

_shift(p::Real, s::Int, nT::Int) = p
_shift(p::Pair{Int,<:Vector}, s::Int, nT::Int) = view(p[2], p[1]+s+1:p[1]+s+nT)
_shift(p::Pair{Int,<:Matrix}, s::Int, nT::Int) = view(p[2], :, p[1]+s+1:p[1]+s+nT)

_getval(p::Real, t::Int) = p
_getval(p::AbstractVector, t::Int) = p[t]
_getval(p::AbstractMatrix, t::Int) = view(p, :, t)

function transition!(varpaths::AbstractDict, b::SimpleBlock, nT::Int)
    inpaths = ((_shift(varpaths[name(v)], shift(v), nT) for v in invars(b))...,)
    outpaths = ((_shift(varpaths[n], 0, nT) for n in outputs(b))...,)
    for t in 1:nT
        vals = b((_getval(p, t) for p in inpaths)...)
        for (i, val) in enumerate(vals)
            if val isa AbstractVector
                outpaths[i][:,t] .= val
            else
                outpaths[i][t] = val
            end
        end
    end
end

function show(io::IO, b::SimpleBlock)
    fname = String(typeof(b).parameters[2].name.name)[2:end]
    print(io, "SimpleBlock($fname)")
end

function _showinouts(io::IO, b::AbstractBlock)
    print(io, "  inputs:  ")
    join(io, invars(b), ", ")
    print(io, "\n  outputs: ")
    join(io, outputs(b), ", ")
end

function show(io::IO, ::MIME"text/plain", b::SimpleBlock)
    fname = String(typeof(b).parameters[2].name.name)[2:end]
    print(io, "SimpleBlock($fname)")
    println(io, hascache(b) ? " with cache:" : ":")
    _showinouts(io, b)
end
