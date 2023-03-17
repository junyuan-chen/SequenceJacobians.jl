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

abstract type AbstractBlockJacobian{TF} end

struct ArrayToArgs{cumwidths}
    function ArrayToArgs(cumwidths::NTuple{N,Int}) where N
        N >= 1 || throw(ArgumentError("length of cumwidths must be at least 1"))
        return new{cumwidths}()
    end
end

function (aa::ArrayToArgs{W})(A::AbstractArray) where W
    if @generated
        i0 = 0
        ex = :()
        for w in W
            i0 < w || error("invalid cumwidths $W")
            # Singleton array input is treated as scalar, which should be fine
            push!(ex.args, i0+1 < w ? :(view(A, $(i0+1):$w)) : :(A[$(i0+1)]))
            i0 = w
        end
        return ex
    else
        i0 = 0
        args = ()
        for w in W
            i0 < w || error("invalid cumwidths $W")
            args = i0+1 < w ? (args..., view(A, i0+1:w)) : (args..., A[i0+1])
            i0 = w
        end
        return args
    end
end

struct PartialF{F<:Function, TF<:Real, CA, I}
    f::F
    inds::Vector{Int}
    vals::Vector{TF}
    cache::CA
    toargs::ArrayToArgs{I}
end

_hascache(::PartialF{F,TF,CA}) where {F,TF,CA} = CA !== Nothing

function (g::PartialF)(fx::AbstractVector, xs)
    length(xs) == length(g.inds) || throw(ArgumentError(
        "expect input xs of length $(length(g.inds)); got $(length(xs))"))
    vals = g.vals
    for (i, x) in zip(g.inds, xs)
        vals[i] = x
    end
    r = _hascache(g) ? g.f(g.cache, g.toargs(vals)...) : g.f(g.toargs(vals)...)
    copyto!(fx, Iterators.flatten(r))
    return fx
end

const PseudoBlockMat{TF} = PseudoBlockMatrix{TF, Matrix{TF}, Tuple{BlockedUnitRange{Vector{Int64}}, BlockedUnitRange{Vector{Int64}}}}

struct SimpleBlockJacobian{BLK<:SimpleBlock, TF, PF<:PartialF, FD} <: AbstractBlockJacobian{TF}
    blk::BLK
    J::PseudoBlockMat{TF}
    x::Vector{TF}
    iins::Vector{Int}
    nT::Int
    g::PF
    fdcache::FD
end

function (j::SimpleBlockJacobian)(varvals::NamedTuple)
    invals = map(k->getfield(varvals, k), inputs(j.blk))
    copyto!(j.g.vals, Iterators.flatten(_hascache(j.g) ? invals[2:end] : invals))
    # A cache should not be reached from any source variable
    copyto!(j.x, Iterators.flatten((invals[i] for i in j.iins)))
    finite_difference_jacobian!(j.J.blocks, j.g, j.x, j.fdcache)
    return j
end

function SimpleBlockJacobian(b::SimpleBlock, iins, invals::Tuple, cache, outwidths,
        nT::Int, TF::Type)
    widths = map(length, invals)
    cumwidths = cumsum(widths)
    toargs = ArrayToArgs(cumwidths)
    ni = cumwidths[end]
    vals = Vector{TF}(undef, ni)
    cuts = (0, cumwidths...)
    iins = collect(iins)
    wiins = cache === nothing ? iins : iins .- 1 # Handle the offset due to skipping cache
    # Collect indices for each individual input value assuming array inputs have fixed length
    inds = Vector{Int}(undef, sum(i->widths[i], wiins))
    i = 1
    for j in wiins
        for k in cuts[j]+1:cuts[j+1]
            inds[i] = k
            i += 1
        end
    end
    g = PartialF(b.f, inds, vals, cache, toargs)
    # Array variables should have existed in varvals
    nii = length(inds)
    no = sum(outwidths)
    J = Matrix{TF}(undef, no, nii)
    fdcache = JacobianCache(Vector{TF}(undef, nii), Vector{TF}(undef, no))
    BJ = PseudoBlockMatrix(J, collect(outwidths), [widths[i] for i in wiins])
    x = collect(TF, Iterators.flatten((invals[i] for i in wiins)))
    return SimpleBlockJacobian(b, BJ, x, iins, nT, g, fdcache)
end

function jacobian(b::SimpleBlock, iins, nT::Int, varvals::NamedTuple, TF::Type=Float64)
    # Assume there is only one cache per block and it is placed in the front
    # (Do not place it at the end as there could be temporal terms added from macros)
    vallast = varvals[inputs(b)[1]]
    hascache = !(vallast isa Union{Real, AbstractArray{<:Real}})
    # Copy the cache to ensure that original values are not mutated
    cache = hascache ? deepcopy(vallast) : nothing
    invals = map(k->getfield(varvals, k), hascache ? inputs(b)[2:end] : inputs(b))
    outwidths = map(k->length(getfield(varvals, k)), outputs(b))
    j = SimpleBlockJacobian(b, iins, invals, cache, outwidths, nT, TF)
    return j(varvals)
end

@inline function getindex(j::SimpleBlockJacobian, r::Int, i::Int)
    v = view(j.J, Block(r, i))
    return Shift([(shift(invars(j.blk)[j.iins[i]]), 0)], [[v]], size(v))
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

function _show_jac_from_to(io::IO, j::AbstractBlockJacobian)
    ins = inputs(j.blk)
    join(io, map(i->ins[i], j.iins), ", ")
    print(io, " → ")
    join(io, outputs(j.blk), ", ")
end

function show(io::IO, j::SimpleBlockJacobian{BLK}) where BLK
    fname = String(BLK.parameters[1].name.name)[2:end]
    print(io, "SimpleBlockJacobian(", fname, ": ")
    _show_jac_from_to(io, j)
    print(io, ')')
end
