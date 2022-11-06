struct VarSpec
    name::Symbol
    shift::Int
end

varspec(name::Symbol; shift::Int=0) = VarSpec(name, shift)
varspec(v::VarSpec) = v

lag(name::Symbol, n=1) = varspec(name, shift=-n)
lead(name::Symbol, n=1) = varspec(name, shift=n)

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

outlength(b::AbstractBlock, varvals::NamedTuple) =
    sum(k->haskey(varvals, k) ? length(varvals[k])::Int : 1, outputs(b))

function outlength(b::AbstractBlock, varvals::NamedTuple, r::Int)
    vo = outputs(b)[r]
    return haskey(varvals, vo) ? length(varvals[vo])::Int : 1
end

function _checkinsouts(ins, outs, ssins)
    length(ins) > 0 || throw(ArgumentError("the inputs of a block cannot be empty"))
    length(outs) > 0 || throw(ArgumentError("the outputs of a block cannot be empty"))
    ssins ⊆ ins || throw(ArgumentError("ssins must be a subset of the block inputs"))
    isempty(intersect(ins, outs)) ||
        throw(ArgumentError("an input cannot be the output of the same block"))
end

struct SimpleBlock{F<:Function,ins,outs,NI} <: AbstractBlock{ins,outs}
    invars::NTuple{NI,VarSpec}
    ssins::Set{Symbol}
    f::F
    function SimpleBlock(ins::NTuple{NI,Symbol}, invars::NTuple{NI,VarSpec},
            ssins::Set{Symbol}, outs::NTuple{NO,Symbol}, f::F) where {NI,NO,F}
        _checkinsouts(ins, outs, ssins)
        return new{F,ins,outs,NI}(invars, ssins, f)
    end
end

function _inout(ins, outs, ssins)
    ins = ins isa Union{Symbol,VarSpec} ? (ins,) : (ins...,)
    outs = outs isa Symbol ? (outs,) : (outs...,)
    ssins isa Union{Symbol,VarSpec} && (ssins = (ssins,))
    # Must do invars before ins
    invars = map(varspec, ins)
    ins = map(name, ins)
    ssins = Set{Symbol}(map(name, ssins))
    outs = map(name, outs)
    return ins, invars, ssins, outs
end

# Allow irrelevant kwargs for @implicit
block(f::Function, ins, outs; ssins=ins, kwargs...) =
    SimpleBlock(_inout(ins, outs, ssins)..., f)

function (b::SimpleBlock)(x...)
    out = b.f(x...)
    out = out isa Tuple ? out : (out,)
    return NamedTuple{outputs(b)}(out)
end

function steadystate!(b::SimpleBlock, varvals::NamedTuple)
    vals = b(map(k->getfield(varvals, k), inputs(b))...)
    return merge(varvals, vals)
end

jacbyinput(::SimpleBlock) = true

function f_partial!(b::SimpleBlock, varvals, fx, x, ::Val{i}) where i
    ins = inputs(b)
    r = b.f(map(k->getfield(varvals, k), ins[1:i-1])..., x,
        map(k->getfield(varvals, k), ins[i+1:length(ins)])...)
    for (k, v) in enumerate(Iterators.flatten(r))
        fx[k] = v
    end
    return fx
end

function jacobian(b::SimpleBlock, Vi::Val{i}, nT::Int, varvals::NamedTuple,
        TF::Type=Float64) where i
    ins = inputs(b)
    val = varvals[ins[i]]
    # Array variables should have existed in varvals
    no = outlength(b, varvals)
    J = Matrix{TF}(undef, no, length(val))
    f!(fx, x) = f_partial!(b, varvals, fx, x, Vi)
    if val isa Real
        finite_difference_gradient!(J, f!, convert(TF,val))
    else
        finite_difference_jacobian!(J, f!, view(val,:))
    end
    return J
end

function getjacmap(b::SimpleBlock, J::Matrix,
        i::Int, ii::Int, r::Int, rr::Int, r0::Int, nT::Int)
    j = J[r0+rr,ii]
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
    fname = String(typeof(b).parameters[1].name.name)[2:end]
    print(io, "SimpleBlock($fname)")
end

function _showinouts(io::IO, b::AbstractBlock)
    print(io, "  inputs:  ")
    join(io, invars(b), ", ")
    print(io, "\n  outputs: ")
    join(io, outputs(b), ", ")
end

function show(io::IO, ::MIME"text/plain", b::SimpleBlock)
    fname = String(typeof(b).parameters[1].name.name)[2:end]
    println(io, "SimpleBlock($fname):")
    _showinouts(io, b)
end
