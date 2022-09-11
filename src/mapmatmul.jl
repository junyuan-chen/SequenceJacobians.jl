const MatOfMap = AbstractMatrix{<:LinearMap}

# A workaround for compositions involving matrices of LinearMaps
# which generate type information that is too long
struct MatMulMap{T, TM<:MatOfMap, LM, RM, AM} <: LinearMap{T}
    A::TM
    lmap::LM
    rmap::RM
    amap::AM
    # lmap, rmap and amap could be a MatOfMap, MatMulMap or Nothing
    function MatMulMap(A::MatOfMap, lmap, rmap, amap=nothing)
        lmap === nothing && rmap === nothing && throw(ArgumentError(
            "lmap and rmap cannot both be nothing"))
        # The eltype of the matrices may be abstract and hence use the first element
        T = eltype(A[1])
        if lmap !== nothing
            size(lmap,2) == size(A,1) || throw(DimensionMismatch())
            T = promote_type(T, lmap isa AbstractMatrix ? eltype(lmap[1]) : eltype(lmap))
        end
        if rmap !== nothing
            size(rmap,1) == size(A,2) || throw(DimensionMismatch())
            T = promote_type(T, rmap isa AbstractMatrix ? eltype(rmap[1]) : eltype(rmap))
        end
        # Check dimension for amap elsewhere
        amap === nothing ||
            (T = promote_type(T, amap isa AbstractMatrix ? eltype(amap[1]) : eltype(amap)))
        return new{T,typeof(A),typeof(lmap),typeof(rmap),typeof(amap)}(A, lmap, rmap, amap)
    end
end

# The size is determined based on the matrices without considering the elements
size(M::MatMulMap{<:Any,<:Any,Nothing}) = (size(M.A,1), size(M.rmap,2))
size(M::MatMulMap{<:Any,<:Any,<:Any,<:Nothing}) = (size(M.lmap,1), size(M.A,2))
size(M::MatMulMap{<:Any,<:Any,<:Any,<:Any}) = (size(M.lmap,1), size(M.rmap,2))

MulStyle(::MatMulMap) = FiveArg()

function _mul!(C::AbstractVecOrMat, A::MatOfMap, B::AbstractVecOrMat, α::Number, β::Number)
    # Assume each LinearMap in A is square with the same size
    nT = size(A[1], 1)
    M = size(A, 1)
    N = size(B, 2)
    K = size(A, 2)
    K * nT == size(B, 1) || throw(DimensionMismatch(
        "matrix A with $(K * nT) columns and matrix B with $(size(B,1)) rows"))
    size(C, 1) == M * nT && size(C, 2) == N ||throw(DimensionMismatch(
        "matrix C has size $(size(C)); expect size ($(M*nT), $N)"))
    for m in 1:M
        rr = 1+(m-1)*nT:m*nT
        mul!(view(C,rr,:), A[m,1], view(B,1:nT,:), α, β)
        if K > 1
            for k in 2:K
                cc = 1+(k-1)*nT:k*nT
                mul!(view(C,rr,:), A[m,k], view(B,cc,:), α, true)
            end
        end
    end
    return C
end

_mul!(C::AbstractVecOrMat, A::MatOfMap, B::AbstractVecOrMat) =
    _mul!(C, A, B, true, zero(eltype(C)))

# Needed for the case when lmap or rmap is a MatMulMap
_mul!(C::AbstractVecOrMat, M::MatMulMap, B::AbstractVecOrMat, α::Number, β::Number) =
    _unsafe_mul!(C, M, B, α, β)

_mul!(C::AbstractVecOrMat, M::MatMulMap, B::AbstractVecOrMat) =
    _unsafe_mul!(C, M, B, true, zero(eltype(C)))

function _unsafe_mul!(C::AbstractVecOrMat{T}, M::MatMulMap, B::AbstractMatrix,
        α::Number, β::Number) where T
    # Assume that each LinearMap is square with the same size
    nT = size(M.A[1], 1)
    if M.rmap !== nothing
        rtemp = Matrix{T}(undef, size(M.rmap,1)*nT, size(B,2))
        _mul!(rtemp, M.rmap, B)
        if M.lmap !== nothing
            temp = Matrix{T}(undef, size(M.A,1)*nT, size(rtemp,2))
            _mul!(temp, M.A, rtemp)
            _mul!(C, M.lmap, temp, α, β)
        else
            _mul!(C, M.A, rtemp, α, β)
        end
    else
        # M.lmap cannot be nothing in this case
        temp = Matrix{T}(undef, size(M.A,1)*nT, size(B,2))
        _mul!(temp, M.A, B)
        _mul!(C, M.lmap, temp, α, β)
    end
    M.amap === nothing || _mul!(C, M.amap, B, α, true)
    return C
end

_unsafe_mul!(C::AbstractVecOrMat, M::MatMulMap, B::AbstractVector, α::Number, β::Number) =
    _unsafe_mul!(C, M, reshape(B, length(B), 1), α, β)

_unsafe_mul!(C::AbstractVecOrMat, M::MatMulMap, B::AbstractVecOrMat) =
    _unsafe_mul!(C, M, B, true, zero(eltype(C)))

_unsafe_mul!(C::AbstractVecOrMat, M::MatMulMap, B::AbstractMatrix) =
    _unsafe_mul!(C, M, B, true, zero(eltype(C)))

# Assume nothing is wrong in the internal of M
function check_dim_mul(C, M::MatMulMap, B)
    size(C, 2) == size(B, 2) || throw(DimensionMismatch(
        "C has size $(size(C)) while B has size $(size(B))"))
    nT = size(M.A[1], 1)
    Kr = M.rmap === nothing ? nT * size(M.A,2) : nT * size(M.rmap,2)
    Kr == size(B, 1) || throw(DimensionMismatch(
        "M does not match the size of B $(size(B))"))
    Kl = M.lmap === nothing ? nT * size(M.A,1) : nT * size(M.lmap,1)
    Kl == size(C, 1) || throw(DimensionMismatch(
        "M does not match the size of C $(size(C))"))
    return nothing
end

(+)(A::MatMulMap, B::MatOfMap) =
    MatMulMap(A.A, A.lmap, A.rmap, A.amap===nothing ? B : A.amap .+ B)

(+)(A::MatOfMap, B::MatMulMap) = B + A

(+)(A::MatMulMap, B::MatMulMap) =
    MatMulMap(A.A, A.lmap, A.rmap, A.amap===nothing ? B : A.amap + B)

mapmatmul(A::MatOfMap, B::AbstractMatrix{<:WrappedMap}) =
    MatMulMap(B, A, nothing)

mapmatmul(A::AbstractMatrix{<:WrappedMap}, B::MatOfMap) =
    MatMulMap(A, nothing, B)

# A has only one column in this case
mapmatmul(A::AbstractMatrix{<:WrappedMap}, B::LinearMap) =
    MatMulMap(A, nothing, reshape([B], 1, 1))

mapmatmul(A::MatMulMap, B::MatOfMap) =
    MatMulMap(A.A, A.lmap, mapmatmul(A.rmap, B))

mapmatmul(A::MatOfMap, B::MatMulMap) =
    MatMulMap(B.A, mapmatmul(A, B.lmap), B.rmap)

mapmatmul(A::MatMulMap, B::MatMulMap) =
    MatMulMap(A.A, A.lmap, mapmatmul(A.rmap, B))

# MatMulMap may contain Nothing
mapmatmul(::Nothing, B::MatOfMap) = B
mapmatmul(A::MatOfMap, ::Nothing) = A
mapmatmul(::Nothing, ::Nothing) = nothing

# Fallback methods that do not use MatMulMap
function mapmatmul(A::MatOfMap, B::MatOfMap)
    T = promote_type(eltype(A[1]), eltype(B[1]))
    out = Matrix{LinearMap{T}}(undef, size(A, 1), size(B, 2))
    mul!(out, A, B)
    return out
end

function mapmatmul(A::MatOfMap, B::LinearMap)
    T = promote_type(eltype(A[1]), eltype(B))
    size(A, 2) == 1 || throw(DimensionMismatch(
        "matrix A with $(size(A,2)) columns and LinearMap B"))
    M = size(A, 1)
    out = Matrix{LinearMap{T}}(undef, M, 1)
    for m in 1:M
        out[m] = A[m,1] * B
    end
    return out
end

function _mat!(C::AbstractVecOrMat, A::MatOfMap, B::MatOfMap)
    # Assume each LinearMap in A is square with the same size
    nT = size(A[1], 1)
    M = size(A, 1)
    N = size(B, 2)
    K = size(A, 2)
    K == size(B, 1) || throw(DimensionMismatch(
        "matrix A with $K columns and matrix B with $(size(B,1)) rows"))
    size(C, 1) == M*nT && size(C, 2) == N*nT ||throw(DimensionMismatch(
        "matrix C has size $(size(C)); expect size ($(M*nT), $(N*nT)"))
    mB = Vector{Matrix{eltype(B[1])}}(undef, K)
    for n in 1:N
        # First convert maps in B to matrices
        if n == 1
            for k in 1:K
                mB[k] = Matrix(B[k,n])
            end
        else
            for k in 1:K
                _unsafe_mul!(mB[k], B[k,n], true)
            end
        end
        for m in 1:M
            tar = view(C,1+(m-1)*nT:m*nT,1+(n-1)*nT:n*nT)
            _unsafe_mul!(tar, A[m,1], mB[1])
            if K > 1
                for k in 2:K
                    _unsafe_mul!(tar, A[m,k], mB[k], true, true)
                end
            end
        end
    end
    return C
end

function Matrix(M::MatMulMap{T}) where T
    nT = size(M.A[1], 1)
    if M.rmap !== nothing
        temp = Matrix{T}(undef, size(M.A,1)*nT, size(M.rmap,2)*nT)
        _mat!(temp, M.A, M.rmap)
        if M.lmap !== nothing
            out = Matrix{T}(undef, size(M.lmap,1)*nT, size(M.A,2)*nT)
            _mul!(out, M.lmap, temp)
        else
            out = temp
        end
    else
        out = Matrix{T}(undef, size(M.lmap,1)*nT, size(M.A,2)*nT)
        _mat!(out, M.lmap, M.A)
    end
    if M.amap !== nothing
        if M.amap isa MatMulMap
            out .+= Matrix(M.amap)
        else
            R, C = size(M.amap)
            for c in 1:C
                for r in 1:R
                    out[1+(r-1)*nT:r*nT,1+(c-1)*nT:c*nT] .+= Matrix(M.amap[r,c])
                end
            end
        end
    end
    return out
end
