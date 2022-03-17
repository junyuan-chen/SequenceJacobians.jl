"""
    linfconverged(A::AbstractArray, B::AbstractArray, tol::Real=1e-8)

Assess convergence by determining whether
the largest absolute difference between corresponding elements in `A` and `B`
is no greater than `tol`.
"""
function linfconverged(A::AbstractArray, B::AbstractArray, tol::Real=1e-8)
    length(A) == length(B) || throw(ArgumentError("the two arrays must have the same length"))
    @inbounds for i in 1:length(A)
        abs(A[i]-B[i]) > tol && return false
    end
    return true
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
function interpolate_y!(yq::AbstractArray, xq::AbstractArray, y::AbstractVector, x::AbstractVector)
    size(yq) == size(xq) || throw(ArgumentError("size of yq must match size of xq"))
    size(y) == size(x) || throw(ArgumentError("size of y must match size of x"))
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
function interpolate_coord!(xqi::AbstractArray, xqpi::AbstractArray, xq::AbstractVector, x::AbstractVector)
    size(xqi) == size(xqpi) == size(xq) ||
        throw(ArgumentError("size of xqi, xqpi and xq must be the same"))
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
    _eachslice(A::AbstractArray)

A variant of `eachslice` for iterating over the last dimension of `A`.
"""
@inline function _eachslice(A::AbstractArray)
    ndim = ndims(A)
    inds_before = ntuple(d->(:), ndim-1)
    return (view(A, inds_before..., i) for i in axes(A, ndim))
end

"""
    _viewslice(A::AbstractArray, i::Int)

View the `i`th slice across the last dimension of `A`.
"""
@inline function _viewslice(A::AbstractArray, i::Int)
    ndim = ndims(A)
    inds_before = ntuple(d->(:), ndim-1)
    return view(A, inds_before..., i)
end

