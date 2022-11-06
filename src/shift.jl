using LinearAlgebra: diagind, rmul!
using LinearMaps: CompositeMap, UniformScalingMap, diagm
using SparseArrays: spdiagm

import Base: ndims, has_offset_axes, copy, iszero, transpose, +, -, *, /, size
import LinearAlgebra: isdiag, mul!
import LinearMaps: LinearMap, MulStyle, FiveArg, _unsafe_mul!

const Tuple2 = Tuple{Int,Int}
const SInd = IdDict{Tuple2,Int}

struct Shift{T<:Number}
    d::SInd
    v::Vector{T}
end

# Sign convention for lead/lag follows the Python package
# The paper appendix uses the opposite sign
Lag(v::Number=1.0) = Shift(SInd((-1,0)=>1), [v])
Lead(v::Number=1.0) = Shift(SInd((1,0)=>1), [v])

Shift(i::Int, v::Number=1.0) = Shift(SInd((i,0)=>1), [v])

function Shift(kvs::Pair{Tuple2,T}...) where T
    d = SInd()
    v = T[]
    for (k, s) in kvs
        i = get(d, k, 0)
        if i === 0
            push!(v, s)
            d[k] = length(v)
        else
            v[i] += s
        end
    end
    return Shift(d, v)
end

function _zdiag(v::T, n::Int, i::Int, m::Int) where T
    out = fill(v, n-abs(i))
    m > 0 && (out[1:m] .= zero(T))
    return out
end

(S::Shift)(n::Integer) =
    spdiagm(n, n, (k[1]=>_zdiag(S.v[i], n, k...) for (k, i) in S.d if n-abs(i)>0)...)

eltype(::Type{Shift{T}}) where T = T
ndims(::Shift) = 2
has_offset_axes(::Shift) = false
copy(S::Shift) = Shift(copy(S.d), copy(S.v))
convert(::Type{Shift{T}}, S::Shift) where T = Shift(S.d, convert(Vector{T}, S.v))

isdiag(S::Shift) = all(k[1]==0 for k in keys(S.d))
iszero(S::Shift) = iszero(S.v)

function transpose(S::Shift)
    d = SInd()
    for (k, i) in S.d
        d[(-k[1],k[2])] = i
    end
    # Values are not copied
    return Shift(d, S.v)
end

function _addorsub(S1::Shift, S2::Shift, op)
    d = copy(S1.d)
    v = copy(S1.v)
    for (k, i) in S2.d
        ind = get(S1.d, k, 0)
        v2 = S2.v[i]
        if ind === 0
            push!(v, op(v2))
            d[k] = length(v)
        else
            v[ind] = op(v[ind], v2)
        end
    end
    return Shift(d, v)
end

function _addorsub(S::Shift, M::AbstractMatrix, op)
    r, c = size(M)
    r == c || throw(ArgumentError("matrix is not square"))
    out = op(M)
    @inbounds for (k, i) in S.d
        id, m = k
        -r < id < r && m < r - abs(id) || continue
        d = view(out, diagind(out, id))
        d[m+1:end] .+= S.v[i]
    end
    return out
end

function _addscale(S1::Shift, sc::Number)
    d = copy(S1.d)
    v = copy(S1.v)
    ind = get(S1.d, (0,0), 0)
    if ind === 0
        push!(v, sc)
        d[(0,0)] = length(v)
    else
        v[ind] += sc
    end
    return Shift(d, v)
end

function _mulind(i::Int, m::Int, j::Int, n::Int)
    k = i + j
    l = ifelse(isless(i, 0),
            ifelse(isless(0, j),
                max(m, n) + min(-i, j),
                max(m+j, n)),
            ifelse(isless(j, 0),
                ifelse(isless(k, 0),
                    max(m+k, n),
                    max(m, n-k)),
                max(m, n-i)))
    return k, l
end

(+)(S::Shift) = copy(S)
(+)(S1::Shift, S2::Shift) = _addorsub(S1, S2, +)
(+)(S::Shift, M::AbstractMatrix) = _addorsub(S, M, +)
(+)(M::AbstractMatrix, S::Shift) = S + M

(-)(S::Shift) = Shift(copy(S.d), -S.v)
(-)(S1::Shift, S2::Shift) = _addorsub(S1, S2, -)
(-)(S::Shift, M::AbstractMatrix) = _addorsub(S, M, -)
(-)(M::AbstractMatrix, S::Shift) = M + (-S)

(*)(S::Shift, x::Number) = Shift(copy(S.d), S.v*x)
(*)(x::Number, S::Shift) = S * x
(/)(S::Shift, x::Number) = Shift(copy(S.d), S.v/x)

function (*)(S1::Shift, S2::Shift)
    d = SInd()
    v = eltype(S1)[]
    for (im, i1) in S1.d
        for (jn, i2) in S2.d
            kl = _mulind(im[1], im[2], jn[1], jn[2])
            p = S1.v[i1] * S2.v[i2]
            ind = get(d, kl, 0)
            if ind === 0
                push!(v, p)
                d[kl] = length(v)
            else
                v[ind] = p
            end
        end
    end
    return Shift(d, v)
end

@inline function mul!(C::AbstractVecOrMat, S::Shift, B::AbstractVecOrMat, α::Number, β::Number)
    # C could contain NaN
    iszero(β) ? fill!(C, zero(eltype(C))) : rmul!(C, β)
    # Need to call size twice in case B is a Vector
    r, c = size(B, 1), size(B, 2)
    @inbounds for (k, i) in S.d
        id, m = k
        -r < id < r && m < r - abs(id) || continue
        v = α * S.v[i]
        # Avoid branching
        adj1 = min(id, 0)
        adj2 = max(id, 0)
        for j in 1:c
            for i in m+1-adj1:r-adj2
                C[i,j] += v * B[i+id,j]
            end
        end
    end
    return C
end

(*)(S::Shift, B::AbstractVecOrMat) = mul!(similar(B), S, B, true, false)

function ==(S1::Shift, S2::Shift)
    length(S1.v) == length(S2.v) || return false
    for (k1, i1) in S1.d
        i2 = get(S2.d, k1, 0)
        i2 === 0 && return false
        S1.v[i1] == S2.v[i2] || return false
    end
    return true
end

struct ShiftMap{T} <: LinearMap{T}
    S::Shift{T}
    N::Int
    function ShiftMap(S::Shift{T}, N::Int) where T
        N > 0 || throw(ArgumentError("size of ShiftMap must be positive"))
        return new{T}(S, N)
    end
end

LinearMap(S::Shift, N::Int) = ShiftMap(S, N)

size(S::ShiftMap) = (S.N, S.N)
MulStyle(::ShiftMap) = FiveArg()
==(S1::ShiftMap, S2::ShiftMap) = S1.S == S2.S && S1.N == S2.N

transpose(S::ShiftMap) = ShiftMap(transpose(S.S), S.N)

_unsafe_mul!(C::AbstractVecOrMat, S::ShiftMap, B::AbstractVector) =
    mul!(C, S.S, B)

_unsafe_mul!(C::AbstractMatrix, S::ShiftMap, B::AbstractMatrix) =
    mul!(C, S.S, B)

_unsafe_mul!(C::AbstractVecOrMat, S::ShiftMap, B::AbstractVector, α::Number, β::Number) =
    mul!(C, S.S, B, α, β)

# Needed for avoiding method ambiguity
_unsafe_mul!(C::AbstractMatrix, S::ShiftMap, B::AbstractMatrix, α::Number, β::Number) =
    mul!(C, S.S, B, α, β)

(+)(S1::ShiftMap, S2::ShiftMap) =
    S1.N==S2.N ? ShiftMap(S1.S+S2.S, S1.N) : throw(DimensionMismatch())
(-)(S1::ShiftMap, S2::ShiftMap) =
    S1.N==S2.N ? ShiftMap(S1.S-S2.S, S1.N) : throw(DimensionMismatch())
(*)(x::Union{Real,Complex}, S::ShiftMap{T}) where T<:Union{Real,Complex} =
    ShiftMap(x*S.S, S.N)
(*)(S::ShiftMap{T}, x::Union{Real,Complex}) where T<:Union{Real,Complex} = x * S

function (*)(S1::ShiftMap, S2::ShiftMap)
    check_dim_mul(S1, S2)
    return ShiftMap(S1.S*S2.S, S1.N)
end

(+)(S::ShiftMap, A::UniformScalingMap) =
    S.N==A.M ? ShiftMap(_addscale(S.S, A.λ), S.N) : throw(DimensionMismatch())

(+)(A::UniformScalingMap, S::ShiftMap) = S + A

function (*)(S::ShiftMap, A::UniformScalingMap)
    S.N==A.M || throw(DimensionMismatch())
    if iszero(A.λ)
        return A
    else
        return ShiftMap(S.S*A.λ, S.N)
    end
end

(*)(A::UniformScalingMap, S::ShiftMap) = S * A

# Avoid generating LinearMaps.LinearCombination
(+)(A1::UniformScalingMap, A2::UniformScalingMap) =
    A1.M==A2.M ? UniformScalingMap(A1.λ+A2.λ, A1.M) : throw(DimensionMismatch())

# Avoid generating LinearMaps.CompositeMap
(*)(A1::UniformScalingMap, A2::UniformScalingMap) =
    A1.M==A2.M ? UniformScalingMap(A1.λ*A2.λ, A1.M) : throw(DimensionMismatch())

function (*)(A::CompositeMap{T}, S::ShiftMap{T}) where T
    Afirst = first(A.maps)
    if Afirst isa ShiftMap
        Afirst.N == S.N || throw(DimensionMismatch())
        A2 = ShiftMap(Afirst.S*S.S, S.N)
        return CompositeMap{T}(tuple(A2, Base.tail(A.maps)...))
    else
        return CompositeMap{T}(tuple(S, A.maps...))
    end
end

function (*)(S::ShiftMap{T}, A::CompositeMap{T}) where T
    Alast = last(A.maps)
    if Alast isa ShiftMap
        Alast.N == S.N || throw(DimensionMismatch())
        A2 = ShiftMap(S.S*Alast.S, S.N)
        return CompositeMap{T}(tuple(Base.front(A.maps)..., A2))
    else
        return CompositeMap{T}(tuple(A.maps..., S))
    end
end

zero(S::LinearMap{T}) where T = LinearMap(UniformScaling(zero(T)), size(S,2))
