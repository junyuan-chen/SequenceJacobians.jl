const ValType{T} = Union{T, Vector{T}} where {T<:Real}

struct VarInput
    name::Symbol
    shift::Int
end

var(name::Symbol; shift::Int=0) = VarInput(name, shift)
var(v::VarInput) = v

lag(name::Symbol, n=1) = var(name, shift=-n)
lead(name::Symbol, n=1) = var(name, shift=n)

name(v::VarInput) = getfield(v, :name)
shift(v::VarInput) = getfield(v, :shift)

name(v::Symbol) = v

convert(::Type{Symbol}, v::VarInput) = v.name

abstract type AbstractBlock end

inputs(b::AbstractBlock) = getfield(b, :ins)
invars(b::AbstractBlock) = getfield(b, :invars)
ssinputs(b::AbstractBlock) = getfield(b, :ssins)
outputs(b::AbstractBlock) = getfield(b, :outs)

function _checkinsouts(ins, outs, ssins)
    length(ins) > 0 || throw(ArgumentError("the inputs of a block cannot be empty"))
    length(outs) > 0 || throw(ArgumentError("the outputs of a block cannot be empty"))
    ssins ⊆ ins || throw(ArgumentError("ssins must be a subset of the block inputs"))
    isempty(intersect(ins, outs)) ||
        throw(ArgumentError("an input cannot be the output of the same block"))
end

struct SimpleBlock{F<:Function} <: AbstractBlock
    ins::Vector{Symbol}
    invars::Vector{VarInput}
    ssins::Set{Symbol}
    outs::Vector{Symbol}
    f::F
    function SimpleBlock(ins::Vector{Symbol}, invars::Vector{VarInput},
            ssins::Set{Symbol}, outs::Vector{Symbol}, f::F) where F
        _checkinsouts(ins, outs, ssins)
        return new{F}(ins, invars, ssins, outs, f)
    end
end

function block(f::Function, ins, outs; ssins=ins)
    ins isa Union{Symbol,VarInput} && (ins = (ins,))
    outs isa Symbol && (outs = (outs,))
    ssins isa Union{Symbol,VarInput} && (ssins = (ssins,))
    invars = var.(ins)
    ins = name.(invars)
    ssins = Set{Symbol}(name.(ssins))
    outs = collect(Symbol, outs)
    return SimpleBlock(ins, invars, ssins, outs, f)
end

(b::SimpleBlock)(x...) = b.f(x...)

function steadystate!(varvals::Dict, b::SimpleBlock)
    vals = b.f((varvals[n] for n in b.ins)...)
    for (i, n) in enumerate(b.outs)
        val = get(varvals, n, nothing)
        val isa Vector ? (val.=vals[i]) : (varvals[n] = vals[i])
    end
end

function jacobian(b::SimpleBlock, i::Int, varvals::Dict{Symbol,ValType{TF}}) where TF
    ins = inputs(b)
    vi = ins[i]
    val = varvals[vi]
    function f(x)
        val isa Vector || (x = x[1])
        xs = (ifelse(k==i, x, varvals[ins[k]]) for k in 1:length(ins))
        return collect(Iterators.flatten(b.f(xs...)))
    end
    x = val isa Vector ? val : [val]
    J = Matrix{TF}(undef, length(b.outs), length(x))
    ForwardDiff.jacobian!(J, f, x)
    return J
end

