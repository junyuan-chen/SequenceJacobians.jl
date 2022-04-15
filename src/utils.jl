"""
    supconverged(A::AbstractArray, B::AbstractArray, tol::Real=1e-8)

Assess convergence by determining whether the largest absolute difference
between corresponding elements in `A` and `B` is no greater than `tol`.
"""
@inline function supconverged(A::AbstractArray, B::AbstractArray, tol::Real=1e-8)
    length(A) == length(B) || throw(ArgumentError("the two arrays must have the same length"))
    @inbounds for i in 1:length(A)
        abs(A[i]-B[i]) > tol && return false
    end
    return true
end

_variance(x::AbstractArray, pr::AbstractArray) = sum(pr.*(x.-sum(pr.*x)).^2)

# From QuantEcon.jl
function _rouwenhorst(p::Real, q::Real, m::Real, Δ::Real, n::Integer)
    if n == 2
        return [m-Δ, m+Δ],  [p 1-p; 1-q q]
    else
        _, θ_nm1 = _rouwenhorst(p, q, m, Δ, n-1)
        θN = p    *[θ_nm1 zeros(n-1, 1); zeros(1, n)] +
             (1-p)*[zeros(n-1, 1) θ_nm1; zeros(1, n)] +
             q    *[zeros(1, n); zeros(n-1, 1) θ_nm1] +
             (1-q)*[zeros(1, n); θ_nm1 zeros(n-1, 1)]

        θN[2:end-1, :] ./= 2

        return range(m-Δ, stop=m+Δ, length=n), θN
    end
end

# From QuantEcon.jl
function gth_solve!(A::Matrix{T}) where T<:Real
    n = size(A, 1)
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = sum(A[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        A[k+1:n, k] /= scale

        for j in k+1:n, i in k+1:n
            A[i, j] += A[i, k] * A[k, j]
        end
    end

    # backsubstitution
    x[n] = 1
    @inbounds for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * A[i, k]
    end

    # normalisation
    x /= sum(x)

    return x
end

Base.@propagate_inbounds function _interpolate(x, xq, i, k, K)
    xi = xq[i]
    xhigh = x[k+1]
    while xhigh < xi && k+1 < K
        k += 1
        xhigh = x[k+1]
    end
    return xi, k, x[k], xhigh
end

"""
    interpolate_y!(yq::AbstractArray, xq::AbstractArray, y::AbstractVector, x::AbstractVector)

Linearly interpolate values evaluated at `xq` for the mapping between `x` and `y`
and store the results in `yq`.
The implementation mostly follows the corresponding method in the original Python package.
See also [`interpolate_coord!`](@ref).
"""
@inline function interpolate_y!(yq::AbstractArray, xq::AbstractArray, y::AbstractVector,
        x::AbstractVector)
    size(yq) == size(xq) || throw(DimensionMismatch("size of yq must match size of xq"))
    size(y) == size(x) || throw(DimensionMismatch("size of y must match size of x"))
    issorted(x) || (x = sort(x))
    isorted = issorted(xq) ? (1:length(xq)) : sortperm(xq)
    K = length(x)
    k = 1
    xlow = x[1]
    xhigh = x[2]
    @inbounds for i in isorted
        xi, k, xlow, xhigh = _interpolate(x, xq, i, k, K)
        p = (xhigh - xi) / (xhigh - xlow)
        yq[i] = p * y[k] + (1-p) * y[k+1]
    end
    return yq
end

"""
    interpolate_coord!(xqi::AbstractArray, xqpi::AbstractArray, xq::AbstractVector, x::AbstractVector)

Store the indices and weights associated with the lower end-points needed
for linear interpolation of `xq` on a grid `x` in `xqi` and `xqpi` respectively.
The implementation mostly follows the corresponding method in the original Python package.
See also [`interpolate_y!`](@ref).
"""
@inline function interpolate_coord!(xqi::AbstractVector, xqpi::AbstractVector,
        xq::AbstractVector, x::AbstractVector)
    length(xqi) == length(xqpi) == length(xq) ||
        throw(DimensionMismatch("length of xqi, xqpi and xq must be the same"))
    issorted(x) || (x = sort(x))
    isorted = issorted(xq) ? (1:length(xq)) : sortperm(xq)
    K = length(x)
    k = 1
    xlow = x[1]
    xhigh = x[2]
    @inbounds for i in isorted
        xi, k, xlow, xhigh = _interpolate(x, xq, i, k, K)
        xqpi[i] = (xhigh - xi) / (xhigh - xlow)
        xqi[i] = k
    end
    return xqi, xqpi
end

"""
    apply_coord!(yq::AbstractArray, y::AbstractArray, li::AbstractArray, lp::AbstractArray)

Set `yq` using indices `li` and weights `lp` of the lower end-points
obtained from interpolataion of `y`.
The implementation mostly follows the corresponding method in the original Python package.
"""
@inline function apply_coord!(yq::AbstractArray, y::AbstractArray, li::AbstractArray,
        lp::AbstractArray, imax=length(yq))
    size(yq) == size(li) == size(lp) ||
        throw(DimensionMismatch("size of yq, li and lp must be the same"))
    @inbounds @simd for i in 1:imax
        lpi, lii = lp[i], li[i]
        yq[i] = lpi * y[lii] + (1-lpi) * y[lii+1]
    end
end

"""
    setmin!(a::AbstractArray, amin::Real)

Replace any element in `a` that is smaller than `amin` with `amin`.
"""
@inline function setmin!(a::AbstractArray, amin::Real)
    amin = convert(eltype(a), amin)
    @inbounds @simd for i in eachindex(a)
        ai = a[i]
        a[i] = ifelse(ai<amin, amin, ai)
    end
end
