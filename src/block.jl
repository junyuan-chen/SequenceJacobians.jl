const ValType{T<:Real} = Union{T, AbstractArray{T}}
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

abstract type AbstractBlock end

inputs(b::AbstractBlock) = getfield(b, :ins)
invars(b::AbstractBlock) = getfield(b, :invars)
ssinputs(b::AbstractBlock) = getfield(b, :ssins)
outputs(b::AbstractBlock) = getfield(b, :outs)

function _countcache(cache::ValidCache, outs::Vector{Symbol})
    count = 0
    for n in outs
        o = get(cache, n, nothing)
        l = o === nothing ? 1 : length(o)
        count += l
    end
    return count
end

hascache(b::AbstractBlock) = hasfield(typeof(b), :cache) && b.cache isa Dict
nouts(b::AbstractBlock) = hascache(b) ? _countcache(b.cache, outputs(b)) : length(outputs(b))

function outlength(b::AbstractBlock, r::Int)
    hascache(b) || return 1
    out = get(b.cache, outputs(b)[r], nothing)
    return out === nothing ? 1 : length(out)
end

function _checkinsouts(ins, outs, ssins)
    length(ins) > 0 || throw(ArgumentError("the inputs of a block cannot be empty"))
    length(outs) > 0 || throw(ArgumentError("the outputs of a block cannot be empty"))
    ssins ⊆ ins || throw(ArgumentError("ssins must be a subset of the block inputs"))
    isempty(intersect(ins, outs)) ||
        throw(ArgumentError("an input cannot be the output of the same block"))
end

struct SimpleBlock{CA<:ValidCache,F<:Function} <: AbstractBlock
    ins::Vector{Symbol}
    invars::Vector{VarSpec}
    ssins::Set{Symbol}
    outs::Vector{Symbol}
    cache::CA
    f::F
    function SimpleBlock(ins::Vector{Symbol}, invars::Vector{VarSpec}, ssins::Set{Symbol},
            outs::Vector{Symbol}, cache::CA, f::F) where {CA,F}
        _checkinsouts(ins, outs, ssins)
        return new{CA,F}(ins, invars, ssins, outs, cache, f)
    end
end

function _inout(ins, outs, ssins)
    ins isa Union{Symbol,VarSpec} && (ins = (ins,))
    outs isa Symbol && (outs = (outs,))
    ssins isa Union{Symbol,VarSpec} && (ssins = (ssins,))
    invars = var.(ins)
    ins = name.(invars)
    ssins = Set{Symbol}(name.(ssins))
    outs = collect(Symbol, outs)
    return ins, invars, ssins, outs
end

block(f::Function, ins, outs; ssins=ins, cache=nothing) =
    SimpleBlock(_inout(ins, outs, ssins)..., cache, f)

hascache(b::SimpleBlock{Nothing}) = false
nouts(b::SimpleBlock{Nothing}) = length(b.outs)
outlength(b::SimpleBlock{Nothing}, r::Int) = 1

function (b::SimpleBlock)(x...)
    out = hascache(b) ? b.f(b.cache, x...) : b.f(x...)
    # This prevents iterating over the elements of a Vector output
    return out isa Tuple ? out : (out,)
end

function steadystate!(b::SimpleBlock, varvals::AbstractDict)
    vals = b((varvals[n] for n in inputs(b))...)
    for (i, n) in enumerate(outputs(b))
        val = get(varvals, n, nothing)
        val isa AbstractArray ? copyto!(val, vals[i]) : (varvals[n] = vals[i])
    end
end

function jacobian(b::SimpleBlock, i::Int, varvals::Dict{Symbol,ValType{TF}}) where TF
    ins = inputs(b)
    vi = ins[i]
    val = varvals[vi]
    function f(x)
        xs = (ifelse(k==i, x, varvals[ins[k]]) for k in 1:length(ins))
        return collect(Iterators.flatten(b.f(xs...)))
    end
    J = Matrix{TF}(undef, nouts(b), length(val))
    if val isa AbstractArray
        ForwardDiff.jacobian!(J, f, vec(val))
    else
        ForwardDiff.derivative!(J, f, val)
    end
    return J
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
