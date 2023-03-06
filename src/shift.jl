const SubMat{T} = SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
const MatOrSub{T} = Union{Matrix{T}, SubMat{T}}

struct Shift{T<:Number, S, M<:MatOrSub{T}}
    d::Vector{Tuple{Int,Int}}
    v::Vector{Vector{M}}
    size::Tuple{Int,Int}
    Shift(d::Vector{Tuple{Int,Int}}, v::Vector{Vector{M}}, size::Tuple{Int,Int}) where M =
        new{eltype(eltype(eltype(v))), size==(1,1), M}(d, v, size)
end

struct CompositeShift{T<:Number, V<:Union{T,Matrix{T}}}
    d::Vector{Tuple{Int,Int}}
    v::Vector{V}
    size::Tuple{Int,Int}
    CompositeShift(d::Vector{Tuple{Int,Int}}, v::Vector{<:Union{T,Matrix{T}}},
        size::Tuple{Int,Int}) where T = new{T,eltype(v)}(d, v, size)
end

const ShiftOrComp{T} = Union{Shift{T}, CompositeShift{T}}

# Sign convention for lead/lag follows the Python package (lags take negative values)
# The paper appendix uses the opposite sign

@inline function mul!(C::AbstractMatrix{T1}, S::Shift{T2,true}, s::Number,
        α::Number=true, β::Number=false) where {T1,T2}
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    @inbounds for ((id, m), vs) in zip(S.d, S.v)
        val = convert(T1, s * α * sum(v->v[1], vs))
        for k in diagind(C, id)[m+1:end]
            C[k] += val
        end
    end
    return C
end

@inline function mul!(C::AbstractMatrix{T1}, S::CompositeShift{T2,T2}, s::Number,
        α::Number=true, β::Number=false) where {T1,T2}
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    @inbounds for ((id, m), v) in zip(S.d, S.v)
        val = convert(T1, s * α * v)
        for k in diagind(C, id)[m+1:end]
            C[k] += val
        end
    end
    return C
end

@inline function mul!(C::AbstractMatrix{T1}, S::Shift{T2,false}, s::Number,
        α::Number=true, β::Number=false) where {T1,T2}
    blocksize(C) == S.size || throw(DimensionMismatch(
        "C has block size ($(blocksize(C))); while ($(S.size)) is expected"))
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    M, N = S.size
    @inbounds for j in 1:N
        for i in 1:M
            blk = view(C, Block(i, j))
            for ((id, m), vs) in zip(S.d, S.v)
                val = convert(T1, s * α * sum(v->v[i,j], vs))
                for k in diagind(blk, id)[m+1:end]
                    blk[k] += val
                end
            end
        end
    end
    return C
end

@inline function mul!(C::AbstractMatrix{T1}, S::CompositeShift{T2,Matrix{T2}}, s::Number,
        α::Number=true, β::Number=false) where {T1,T2}
    blocksize(C) == S.size || throw(DimensionMismatch(
        "C has block size ($(blocksize(C))); while ($(S.size)) is expected"))
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    M, N = S.size
    @inbounds for j in 1:N
        for i in 1:M
            blk = view(C, Block(i, j))
            for ((id, m), v) in zip(S.d, S.v)
                val = convert(T1, s * α * v[i,j])
                for k in diagind(blk, id)[m+1:end]
                    blk[k] += val
                end
            end
        end
    end
    return C
end

(S::Shift{T,true})(n::Integer) where T = mul!(zeros(T, n.*S.size), S, true, true, true)
(S::CompositeShift{T,T})(n::Integer) where T = mul!(zeros(T, n.*S.size), S, true, true, true)

function (S::Shift{T,false})(n::Integer) where T
    M, N = S.size
    out = PseudoBlockMatrix(zeros(T, n*M, n*N), Fill(n, M), Fill(n, N))
    return mul!(out, S, true, true, true)
end

function (S::CompositeShift{T,Matrix{T}})(n::Integer) where T
    M, N = S.size
    out = PseudoBlockMatrix(zeros(T, n*M, n*N), Fill(n, M), Fill(n, N))
    return mul!(out, S, true, true, true)
end

eltype(::Type{<:ShiftOrComp{T}}) where T = T
ndims(::ShiftOrComp) = 2
has_offset_axes(::ShiftOrComp) = false
isdiag(S::ShiftOrComp) = all(x->iszero(x[1]), S.d)
iszero(S::Shift) = all(x->all(iszero, x), S.v)
iszero(S::CompositeShift) = all(iszero, S.v)

# Lazy summation that keeps the original value arrays
# Should only be used for summing across temporal terms
@inline function (+)(S1::Shift{T}, S2::Shift{T}) where T
    S1.size == S2.size || throw(DimensionMismatch())
    D = copy(S1.d)
    V = [copy(v) for v in S1.v]
    for (i, d) in enumerate(S2.d)
        k = findfirst(x->x==d, D)
        if k === nothing
            push!(D, d)
            push!(V, copy(S2.v[i]))
        else
            append!(V[k], S2.v[i])
        end
    end
    return Shift(D, V, S1.size)
end

(+)(S::Shift) = S

function (*)(S::Shift{T,true}, s::Number) where T
    D = copy(S.d)
    V = [s * sum(v->v[1], vs) for vs in S.v]
    return CompositeShift(D, V, S.size)
end

function (*)(S::Shift{T,false}, s::Number) where T
    D = copy(S.d)
    V = [rmul!(sum(vs), s) for vs in S.v]
    return CompositeShift(D, V, S.size)
end

(*)(s::Number, S::Shift) = S * s

@inline function _mulind(i::Int, m::Int, j::Int, n::Int)
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

@inline function mul!(S::CompositeShift{T,T}, S1::Shift{T,true},
        S2::CompositeShift{T,T}, α::Number, β::Number) where T
    D = S.d
    V = S.v
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                push!(V, α * sum(v->v[1], S1.v[i1]) * S2.v[i2])
            else
                V[k] = β * V[k] + α * sum(v->v[1], S1.v[i1]) * S2.v[i2]
            end
        end
    end
    return S
end

@inline function mul!(S::CompositeShift{T,T}, S1::Shift{T,true}, s::Number,
        α::Number, β::Number) where T
    D = S.d
    V = S.v
    for (i1, d) in enumerate(S1.d)
        k = findfirst(x->x==d, D)
        if k === nothing
            push!(D, d)
            push!(V, α * sum(v->v[1], S1.v[i1]) * s)
        else
            V[k] = β * V[k] + α * sum(v->v[1], S1.v[i1]) * s
        end
    end
    return S
end

@inline function mul!(S::CompositeShift{T,Matrix{T}}, S1::Shift{T,false},
        S2::CompositeShift{T,Matrix{T}}, α::Number, β::Number) where T
    D = S.d
    V = S.v
    M = S1.size[1]
    N = S2.size[2]
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                m = zeros(T, M, N)
                for l in 1:length(S1.v)
                    mul!(m, S1.v[i1][l], S2.v[i2], α, true)
                end
                push!(V, m)
            else
                m = V[k]
                mul!(m, S1.v[i1][l], S2.v[i2], α, β)
                for l in 2:length(S1.v)
                    mul!(m, S1.v[i1][l], S2.v[i2], α, true)
                end
            end
        end
    end
    return S
end

@inline function mul!(S::CompositeShift{T,Matrix{T}}, S1::Shift{T,false}, s::Number,
        α::Number, β::Number) where T
    D = S.d
    V = S.v
    M = S1.size[1]
    N = S2.size[2]
    for (i1, d) in enumerate(S1.d)
        k = findfirst(x->x==d, D)
        if k === nothing
            push!(D, d)
            m = zeros(T, M, N)
            for l in 1:length(S1.v)
                mul!(m, S1.v[i1][l], s, α, true)
            end
            push!(V, m)
        else
            m = V[k]
            mul!(m, S1.v[i1][l], s, α, β)
            for l in 2:length(S1.v)
                mul!(m, S1.v[i1][l], s, α, true)
            end
        end
    end
    return S
end

# ! To do: consider cases between row/column vector and scalar
(*)(S1::Shift{T,true}, S2::CompositeShift{T,T}) where T =
    mul!(CompositeShift(Tuple{Int,Int}[], T[], (1,1)), S1, S2, true, true)

(*)(S1::Shift{T,false}, S2::CompositeShift{T,Matrix{T}}) where T =
    mul!(CompositeShift(Tuple{Int,Int}[], Matrix{T}[], (S1.size[1],S2.size[2])),
        S1, S2, true, true)

@inline function mul!(C::AbstractVecOrMat{T1}, S::Shift{T2,true}, B::AbstractVecOrMat,
        α::Number, β::Number) where {T1,T2}
    # C could contain NaN
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    # Need to call size twice in case B is a Vector
    r, c = size(B, 1), size(B, 2)
    @inbounds for (k, (id, m)) in enumerate(S.d)
        -r < id < r && m < r - abs(id) || continue
        v = convert(T1, α * sum(v->v[1], S.v[k]))
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

@inline function mul!(C::AbstractVecOrMat{T1}, S::CompositeShift{T2,T2}, B::AbstractVecOrMat,
        α::Number, β::Number) where {T1,T2}
    # C could contain NaN
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    # Need to call size twice in case B is a Vector
    r, c = size(B, 1), size(B, 2)
    @inbounds for (k, (id, m)) in enumerate(S.d)
        -r < id < r && m < r - abs(id) || continue
        v = convert(T1, α * S.v[k])
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

@inline function mul!(C::AbstractVecOrMat{T1}, S::Shift{T2,false}, B::AbstractVecOrMat,
        α::Number, β::Number) where {T1,T2}
    Mc, Nc = blocksize(C)
    M, N = S.size
    # Need to call size twice in case B is a Vector
    Mb, Nb = blocksize(B, 1), blocksize(B, 2)
    Mc == S.size[1] && Nc == Nb && Mb == S.size[2] || throw(DimensionMismatch())
    # C could contain NaN
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    @inbounds for j in 1:N
        for jb in 1:Nb
            Bblk = view(B, Block(j, jb))
            r, c = size(Bblk, 1), size(Bblk, 2)
            for i in 1:M
                for (k, (id, m)) in enumerate(S.d)
                    -r < id < r && m < r - abs(id) || continue
                    v = convert(T1, α * sum(v->v[i,j], S.v[k]))
                    adj1 = min(id, 0)
                    adj2 = max(id, 0)
                    for jc in 1:Nc
                        Cblk = view(C, Block(is, jc))
                        for jj in 1:c
                            for ii in m+1-adj1:r-adj2
                                Cblk[ii,jj] += v * Bblk[ii+id,jj]
                            end
                        end
                    end
                end
            end
        end
    end
    return C
end

@inline function mul!(C::AbstractVecOrMat{T1}, S::CompositeShift{T2,Matrix{T2}},
        B::AbstractVecOrMat, α::Number, β::Number) where {T1,T2}
    Mc, Nc = blocksize(C)
    M, N = S.size
    # Need to call size twice in case B is a Vector
    Mb, Nb = blocksize(B, 1), blocksize(B, 2)
    Mc == S.size[1] && Nc == Nb && Mb == S.size[2] || throw(DimensionMismatch())
    # C could contain NaN
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    @inbounds for j in 1:N
        for jb in 1:Nb
            Bblk = view(B, Block(j, jb))
            r, c = size(Bblk, 1), size(Bblk, 2)
            for i in 1:M
                for (k, (id, m)) in enumerate(S.d)
                    -r < id < r && m < r - abs(id) || continue
                    v = convert(T1, α * S.v[k][i,j])
                    adj1 = min(id, 0)
                    adj2 = max(id, 0)
                    for jc in 1:Nc
                        Cblk = view(C, Block(is, jc))
                        for jj in 1:c
                            for ii in m+1-adj1:r-adj2
                                Cblk[ii,jj] += v * Bblk[ii+id,jj]
                            end
                        end
                    end
                end
            end
        end
    end
    return C
end

(*)(S::Union{<:Shift{T1,true}, CompositeShift{T1,T1}},
    B::AbstractVecOrMat{T2}) where {T1,T2} =
        mul!(zeros(promote_type(T1,T2), size(B)), S, B, true, true)

function (*)(S::Union{<:Shift{T1,false}, CompositeShift{T1,Matrix{T1}}},
        B::AbstractVecOrMat{T2}) where {T1,T2}
    M = S.size[1]
    T = promote_type(T1,T2)
    bsizes = blocksizes(B)
    n = bsizes[1][1]
    all(==(n), bsizes[1]) && all(==(n), bsizes[2]) || throw(ArgumentError(
        "each block in B must have the same size"))
    out = PseudoBlockMatrix(zeros(T, n*M, size(B,2)), Fill(n, M), Fill(n, blocksize(B,2)))
    return mul!(out, S, B, true, true)
end
