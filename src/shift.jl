const SubMat{T} = SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
const MatOrSub{T} = Union{Matrix{T}, SubMat{T}}

struct Shift{T<:Number, S, M<:MatOrSub{T}}
    d::Vector{Tuple{Int,Int}}
    r::Vector{Int}
    p::Vector{Int}
    v::Vector{Vector{M}}
    size::Tuple{Int,Int}
    Shift(d::Vector{Tuple{Int,Int}}, v::Vector{Vector{M}}, size::Tuple{Int,Int}) where M =
        new{eltype(eltype(eltype(v))), size==(1,1), M}(d, Int[], Int[], v, size)
end

struct CompositeShift{T<:Number, V<:Union{T,Matrix{T}}}
    d::Vector{Tuple{Int,Int}}
    r::Vector{Int}
    p::Vector{Int}
    v::Vector{V}
    size::Tuple{Int,Int}
    CompositeShift(d::Vector{Tuple{Int,Int}}, v::Vector{<:Union{T,Matrix{T}}},
        size::Tuple{Int,Int}) where T = new{T,eltype(v)}(d, Int[], Int[], v, size)
end

const ShiftOrComp{T} = Union{Shift{T}, CompositeShift{T}}

# Sign convention for lead/lag follows the Python package (lags take negative values)
# The paper appendix uses the opposite sign

@inline _val(::Shift{T,true}, s::Number, α::Number, vs) where T =
    convert(T, s * α * sum(Fix2(getindex, 1), vs))
@inline _val(::CompositeShift{T,T}, s::Number, α::Number, v) where T =
    convert(T, s * α * v)

@inline function mul!(C::AbstractMatrix{T1},
        S::Union{Shift{T2,true}, CompositeShift{T2,T2}}, s::Number,
        α::Number=true, β::Number=false) where {T1,T2}
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    @inbounds for ((id, m), v) in zip(S.d, S.v)
        val = _val(S, s, α, v)
        for k in diagind(C, id)[m+1:end]
            C[k] += val
        end
    end
    return C
end

@propagate_inbounds _val(::Shift{T,false}, s::Number, α::Number, vs,
    i::Int, j::Int) where T = convert(T, s * α * sum(v->v[i,j], vs))
@propagate_inbounds _val(::CompositeShift{T,Matrix{T}}, s::Number, α::Number, v,
    i::Int, j::Int) where T = convert(T, s * α * v[i,j])

@inline function mul!(C::AbstractMatrix{T1},
        S::Union{Shift{T2,false}, CompositeShift{T2,Matrix{T2}}}, s::Number,
        α::Number=true, β::Number=false) where {T1,T2}
    nT, nT1 = Int.(size(C)./S.size)
    nT == nT1 || throw(DimensionMismatch(
        "C of size ($(size(C))) does not match S of block size ($(S.size))"))
    Cb = _block2(C, nT)
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    M, N = S.size
    @inbounds for j in 1:N
        for i in 1:M
            blk = view(Cb, Block(i, j))
            for ((id, m), v) in zip(S.d, S.v)
                val = _val(S, s, α, v, i, j)
                for k in diagind(blk, id)[m+1:end]
                    blk[k] += val
                end
            end
        end
    end
    return C
end

(S::Union{Shift{T}, CompositeShift{T}})(n::Integer) where T =
    mul!(zeros(T, n.*S.size), S, true, true, true)

function _rankdiag!(r::Vector{Int}, d::Vector{Tuple{Int,Int}}, p::Vector{Int})
    S = length(d)
    resize!(r, S)
    resize!(p, S)
    id = SVector{S}(ntuple(i->d[i][1], S))
    sortperm!(p, id; rev=true)
    _denserank!(r, id, p)
end

# Unsafe because C is assumed to have the correct sparsity pattern
@inline function _unsafe_mul!(C::AbstractSparseMatrixCSC{T1},
        S::Union{Shift{T2,true}, CompositeShift{T2,T2}}, s::Number,
        α::Number=true, β::Number=false; rankdiag::Bool=true) where {T1,T2}
    Mc, Nc = size(C)
    Mc == Nc || throw(ArgumentError("C is expected to be a square matrix"))
    rankdiag && _rankdiag!(S.r, S.d, S.p)
    ptr = getcolptr(C)
    nzs = getnzval(C)
    iszero(β) ? fill!(nzs, zero(T1)) : isone(β) ? nzs : rmul!(nzs, β)
    R = S.r[S.p[end]]
    J1 = max(S.d[S.p[1]][1], 0)
    J2 = max(-S.d[S.p[end]][1], 0)
    @inbounds for (k, v) in enumerate(S.v)
        id, m = S.d[k]
        j1 = max(id,0) + m + 1 # The first column the value appears
        offset = S.r[k] - 1
        val = _val(S, s, α, v)
        # The entire diagonal is structurally non-zero, no matter what m takes
        for j in j1:J1
            # Compute index from backward as the beginning values may not be there
            nzs[ptr[j+1]+offset-R] += val
        end
        for j in max(Nc-J2+1,j1):Nc+min(id,0)
            nzs[ptr[j]+offset] += val
        end
        nzs[ptr[max(J1+1,j1)]+offset:R:ptr[max(Nc-J2+1,j1)]-1] .+= val
    end
    return C
end

@inline function _unsafe_mul!(C::AbstractSparseMatrixCSC{T1},
        S::Union{Shift{T2,false}, CompositeShift{T2,Matrix{T2}}}, s::Number,
        α::Number=true, β::Number=false; rankdiag::Bool=true) where {T1,T2}
    Mc, Nc = size(C)
    nT, nT1 = Int.((Mc, Nc)./S.size)
    nT == nT1 || throw(DimensionMismatch(
        "C of size ($(size(C))) does not match S of block size ($(S.size))"))
    rankdiag && _rankdiag!(S.r, S.d, S.p)
    ptr = getcolptr(C)
    nzs = getnzval(C)
    iszero(β) ? fill!(nzs, zero(T1)) : isone(β) ? nzs : rmul!(nzs, β)
    R = S.r[S.p[end]]
    J1 = max(S.d[S.p[1]][1], 0)
    J2 = max(-S.d[S.p[end]][1], 0)
    M, N = S.size
    @inbounds for bj in 1:N
        for bi in 1:M
            for (k, v) in enumerate(S.v)
                id, m = S.d[k]
                j1 = max(id,0) + m + 1 # The first column the value appears
                offset = S.r[k] - 1
                val = _val(S, s, α, v, bi, bj)
                j0 = (bj-1) * nT
                for j in j0+j1:j0+J1
                    nz = (ptr[j+1] - ptr[j]) ÷ M
                    # Compute index from backward as the beginning values may not be there
                    nzs[ptr[j+1]+offset-R-(M-bi)*nz] += val
                end
                for j in j0+max(nT-J2+1,j1):j0+nT+min(id,0)
                    nz = (ptr[j+1] - ptr[j]) ÷ M
                    nzs[ptr[j]+offset+(bi-1)*nz] += val
                end
                nzs[ptr[j0+max(J1+1,j1)]+(bi-1)*R+offset:M*R:ptr[j0+max(nT-J2+1,j1)]-1] .+= val
            end
        end
    end
    return C
end

function sparse(S::Union{Shift{T}, CompositeShift{T}}, n::Integer) where T
    ids = unique!(getindex.(S.d, 1))
    R = maximum(abs, ids)
    n > R || throw(ArgumentError("requested size of array ($n) is too small"))
    M, N = S.size
    out = spdiagm(n, n, (d => zeros(T, n - abs(d)) for d in ids)...)
    M*N > 1 && (out = hvcat(N, (out for _ in 1:M*N)...))
    return _unsafe_mul!(out, S, true, true, true)
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
    V = Matrix{T}[]
    for vs in S.v
        v1 = rmul!(collect(vs[1]), s)
        for i in 2:length(vs)
            mul!(v1, vs[i], s, true, true)
        end
        push!(V, v1)
    end
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
    # This ensures all values get multiplied
    iszero(β) ? fill!(V, zero(T)) : isone(β) ? V : rmul!(V, β)
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                push!(V, α * sum(v->v[1], S1.v[i1]) * S2.v[i2])
            else
                V[k] += α * sum(v->v[1], S1.v[i1]) * S2.v[i2]
            end
        end
    end
    return S
end

@inline function mul!(S::CompositeShift{T,T}, S1::Shift{T,true}, s::Number,
        α::Number, β::Number) where T
    D = S.d
    V = S.v
    # This ensures all values get multiplied
    iszero(β) ? fill!(V, zero(T)) : isone(β) ? V : rmul!(V, β)
    for (i1, d) in enumerate(S1.d)
        k = findfirst(x->x==d, D)
        if k === nothing
            push!(D, d)
            push!(V, α * sum(v->v[1], S1.v[i1]) * s)
        else
            V[k] += α * sum(v->v[1], S1.v[i1]) * s
        end
    end
    return S
end

@inline function mul!(S::CompositeShift{T,Matrix{T}}, S1::Shift{T,false},
        S2::CompositeShift{T,Matrix{T}}, α::Number, β::Number) where T
    D = S.d
    V = S.v
    # This ensures all values get multiplied
    iszero(β) ? foreach(v->fill!(v, zero(T)), V) : isone(β) ? V : foreach(v->rmul!(v, β), V)
    M = S1.size[1]
    N = S2.size[2]
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                m = zeros(T, M, N)
                for v in S1.v[i1]
                    mul!(m, v, S2.v[i2], α, true)
                end
                push!(V, m)
            else
                m = V[k]
                for v in S1.v[i1]
                    mul!(m, v, S2.v[i2], α, true)
                end
            end
        end
    end
    return S
end

# The case where scalar input meets array output
@inline function mul!(S::CompositeShift{T,Matrix{T}}, S1::Shift{T,true},
        S2::CompositeShift{T,Matrix{T}}, α::Number, β::Number) where T
    D = S.d
    V = S.v
    # This ensures all values get multiplied
    iszero(β) ? foreach(v->fill!(v, zero(T)), V) : isone(β) ? V : foreach(v->rmul!(v, β), V)
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                push!(V, α * sum(v->v[1], S1.v[i1]) * S2.v[i2])
            else
                m = V[k]
                mul!(m, S2.v[i2], α * sum(v->v[1], S1.v[i1]), true, true)
            end
        end
    end
    return S
end

# The case where array input meets scalar output
@inline function mul!(S::CompositeShift{T,Matrix{T}}, S1::Shift{T,false},
        S2::CompositeShift{T,T}, α::Number, β::Number) where T
    M, N = S1.size
    N == 1 || throw(DimensionMismatch("S1 is expected to have only one input variable"))
    D = S.d
    V = S.v
    # This ensures all values get multiplied
    iszero(β) ? foreach(v->fill!(v, zero(T)), V) : isone(β) ? V : foreach(v->rmul!(v, β), V)
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                m = zeros(T, M, N)
                for v in S1.v[i1]
                    m .+= v
                end
                m .*= α * S2.v[i2]
                push!(V, m)
            else
                m = V[k]
                v2 = S2.v[i2]
                for v in S1.v[i1]
                    m .+= α .* v .* v2
                end
            end
        end
    end
    return S
end

# The case where array input degenerates to scalar
@inline function mul!(S::CompositeShift{T,T}, S1::Shift{T,false},
        S2::CompositeShift{T,Matrix{T}}, α::Number, β::Number) where T
    D = S.d
    V = S.v
    # This ensures all values get multiplied
    iszero(β) ? fill!(V, zero(T)) : isone(β) ? V : rmul!(V, β)
    M = S1.size[1]
    N = S2.size[2]
    M == N == 1 || throw(DimensionMismatch("output consists of multiple blocks"))
    for (i1, (id1, m1)) in enumerate(S1.d)
        for (i2, (id2, m2)) in enumerate(S2.d)
            kl = _mulind(id1, m1, id2, m2)
            k = findfirst(x->x==kl, D)
            if k === nothing
                push!(D, kl)
                push!(V, α * sum(v->dot(v, S2.v[i2]), S1.v[i1]))
            else
                V[k] += α * sum(v->dot(v, S2.v[i2]), S1.v[i1])
            end
        end
    end
    return S
end

@inline function mul!(S::CompositeShift{T,Matrix{T}}, S1::Shift{T,false}, s::Number,
        α::Number, β::Number) where T
    D = S.d
    V = S.v
    # This ensures all values get multiplied
    iszero(β) ? foreach(v->fill!(v, zero(T)), V) : isone(β) ? V : foreach(v->rmul!(v, β), V)
    M, N = S1.size
    for (i1, d) in enumerate(S1.d)
        k = findfirst(x->x==d, D)
        if k === nothing
            push!(D, d)
            m = zeros(T, M, N)
            for v in S1.v[i1]
                mul!(m, v, s, α, true)
            end
            push!(V, m)
        else
            m = V[k]
            for v in S1.v[i1]
                mul!(m, v, s, α, true)
            end
        end
    end
    return S
end

(*)(S1::Shift{T,true}, S2::CompositeShift{T,T}) where T =
    mul!(CompositeShift(Tuple{Int,Int}[], T[], (1,1)), S1, S2, true, true)

(*)(S1::Shift{T,false}, S2::CompositeShift{T,T}) where T =
    mul!(CompositeShift(Tuple{Int,Int}[], Matrix{T}[], (S1.size[1],1)), S1, S2, true, true)

(*)(S1::Shift{T,true}, S2::CompositeShift{T,Matrix{T}}) where T =
    mul!(CompositeShift(Tuple{Int,Int}[], Matrix{T}[], (1,S2.size[2])), S1, S2, true, true)

function (*)(S1::Shift{T,false}, S2::CompositeShift{T,Matrix{T}}) where T
    out = S1.size[1] == S2.size[2] == 1 ?
        CompositeShift(Tuple{Int,Int}[], T[], (1,1)) :
            CompositeShift(Tuple{Int,Int}[], Matrix{T}[], (S1.size[1],S2.size[2]))
    return mul!(out, S1, S2, true, true)
end

@propagate_inbounds _getval(S::Shift{T,true}, k::Int, i::Int=0, j::Int=0) where T =
    sum(Fix2(getindex, 1), S.v[k])
@propagate_inbounds _getval(S::CompositeShift{T,T}, k::Int, i::Int=0, j::Int=0) where T =
    S.v[k]
@propagate_inbounds _getval(S::Shift{T,false}, k::Int, i::Int, j::Int) where T =
    sum(v->v[i,j], S.v[k])
@propagate_inbounds _getval(S::CompositeShift{T,Matrix{T}}, k::Int, i::Int,
    j::Int) where T = S.v[k][i,j]

function _shift!(C::AbstractVecOrMat{T1}, S, B, α, rb, rc, c, i, j) where T1
    for (k, (id, m)) in enumerate(S.d)
        -rb < id < rb && m < rb - abs(id) || continue
        v = convert(T1, α * _getval(S, k, i, j))
        adj1 = min(id, 0)
        adj2 = max(id, 0)
        i0 = m + 1 - adj1
        i0 > rc && continue
        il = min(rb-adj2, rc)
        for jj in 1:c
            for ii in i0:il
                @inbounds C[ii,jj] += v * B[ii+id,jj]
            end
        end
    end
    return nothing
end

@inline function mul!(C::AbstractVecOrMat{T1},
        S::Union{Shift{T2,true}, CompositeShift{T2,T2}}, B::AbstractVecOrMat,
        α::Number, β::Number) where {T1,T2}
    # C could contain NaN
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    # B and C are allowed to have different sizes
    rc, rb = size(C, 1), size(B, 1)
    c = min(size(C, 2), size(B, 2))
    _shift!(C, S, B, α, rb, rc, c, 0, 0)
    return C
end

@inline function mul!(C::AbstractVecOrMat{T1},
        S::Union{Shift{T2,false}, CompositeShift{T2,Matrix{T2}}}, B::AbstractVecOrMat,
        α::Number, β::Number) where {T1,T2}
    M, N = S.size
    # B and C are allowed to have different sizes
    rc = Int(size(C,1)/M)
    rb = Int(size(B,1)/N)
    Cb = _block1(C, rc)
    Bb = _block1(B, rb)
    c = min(size(C, 2), size(B, 2))
    # C could contain NaN
    iszero(β) ? fill!(C, zero(T1)) : isone(β) ? C : rmul!(C, β)
    for j in 1:N
        Bblk = view(Bb, Block(j, 1))
        Threads.@threads for i in 1:M # Must put @threads here to get correct results
            Cblk = view(Cb, Block(i, 1))
            _shift!(Cblk, S, Bblk, α, rb, rc, c, i, j)
        end
    end
    return C
end

(*)(S::Union{<:Shift{T1,true}, CompositeShift{T1,T1}},
    B::AbstractVecOrMat{T2}) where {T1,T2} =
        mul!(zeros(promote_type(T1,T2), size(B)), S, B, true, true)

function (*)(S::Union{<:Shift{T1,false}, CompositeShift{T1,Matrix{T1}}},
        B::AbstractVecOrMat{T2}) where {T1,T2}
    nT = Int(size(B,2)/S.size[2])
    return mul!(zeros(promote_type(T1,T2), nT*S.size[1], size(B,2)), S, B, true, true)
end
