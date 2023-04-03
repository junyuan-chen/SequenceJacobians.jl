"""
    AbstractAllCovCache

Supertype for objects used to store intermediate results
needed for computing variance-covariance matrices across all relative time.
"""
abstract type AbstractAllCovCache end

show(io::IO, ca::AbstractAllCovCache) = print(io, typeof(ca).name.name)

"""
    FFTWAllCovCache{TF<:AbstractFloat, P<:Plan, PI<:Plan} <: AbstractAllCovCache

Temporary arrays and `FFTW` plans needed for computing the variance-covariance matrices
across all relative time using fast Fourier transforms.
"""
struct FFTWAllCovCache{TF<:AbstractFloat, P<:Plan, PI<:Plan} <: AbstractAllCovCache
    xpadded::Array{TF,3}
    dft::Array{Complex{TF},3}
    r::Array{Complex{TF},3}
    ir::Array{TF,3}
    plan::P
    iplan::PI
end

"""
    FFTWAllCovCache(T::Int, O::Int, Z::Int, TF::Type=Float64; kwargs...)

Construct a `FFTWAllCovCache` for a vector moving-average process of order `T`-1
consisting of `O` variables with scalar values of type `TF` that are driven by `Z` shocks.
Any keyword argument specified is passed to `FFTW.plan_rfft` and `FFTW.plan_irfft`
for planning the fast Fourier transforms.
"""
function FFTWAllCovCache(T::Int, O::Int, Z::Int, TF::Type=Float64; kwargs...)
    Tfull = 2*T - 2
    xpadded = zeros(TF, Tfull, O, Z)
    dft = Array{Complex{TF},3}(undef, T, O, Z)
    r = Array{Complex{TF},3}(undef, T, O, O)
    ir = Array{TF,3}(undef, Tfull, O, O)
    plan = plan_rfft(xpadded, 1; kwargs...)
    iplan = plan_irfft(r, Tfull, 1; kwargs...)
    return FFTWAllCovCache(xpadded, dft, r, ir, plan, iplan)
end

const _allcovargdoc = """
    The three axes of `x` correspond to time relative to shocks
    (starting from the period on impact), observable outcomes and shocks.
    The optional `σ` can be either a `Real` or a vector of `Real`s
    for the standard errors of the shocks.
    (In the former case, all shock standard errors take the same value.)"""

"""
    allcov!([out], ca::FFTWAllCovCache, x::AbstractArray{<:Real,3}, σ=1)

Compute variance-covariance matrices across all relative time
for the vector moving-average process represented by coefficients in `x`
using `ca` to avoid memory allocations.
Results are copied to `out` if provided
or otherwise returned as a view of `ca.ir`.
This method is for scenarios in which computation for `x` of the same size
needs to be repeated for many times.
See also [`FFTWAllCovCache`](@ref), [`allcov`](@ref) and [`allcor!`](@ref).

$_allcovargdoc
"""
function allcov!(ca::FFTWAllCovCache, x::AbstractArray{<:Real,3},
        σ::Union{Real, AbstractVector{<:Real}}=1)
    T = size(x, 1)
    size(x) == size(ca.dft) || throw(DimensionMismatch(
        "size of x ($(size(x))) is expected to be $(size(ca.dft)) to match ca"))
    copyto!(view(ca.xpadded,1:T,:,:), x)
    mul!(ca.dft, ca.plan, ca.xpadded)
    # threads=false avoids allocation and could even make it slightly faster
    if σ isa Real
        @tullio ca.r[t,o1,o2] = conj(ca.dft[t,o1,z]) * σ^2 * ca.dft[t,o2,z] threads=false
    else
        @tullio ca.r[t,o1,o2] = conj(ca.dft[t,o1,z]) * σ[z]^2 * ca.dft[t,o2,z] threads=false
    end
    mul!(ca.ir, ca.iplan, ca.r)
    return view(ca.ir,1:T,:,:)
end

allcov!(out::AbstractArray, ca::FFTWAllCovCache, x::AbstractArray{<:Real,3}, σ=1) =
    copyto!(out, allcov!(ca, x, σ))

"""
    allcov(x::AbstractArray{<:Real,3}, σ::Union{Real, AbstractVector{<:Real}}=1)

Compute variance-covariance matrices across all relative time
for the vector moving-average process represented by coefficients in `x`.
See also [`allcov!`](@ref) and [`allcor`](@ref).

$_allcovargdoc
"""
function allcov(x::AbstractArray{<:Real,3}, σ::Union{Real, AbstractVector{<:Real}}=1)
    T, O, Z = size(x)
    Tfull = 2*T - 2
    TF = promote_type(eltype(x), σ isa Real ? typeof(σ) : eltype(σ))
    r = Array{Complex{TF},3}(undef, T, O, O)
    # Pad x with zeros
    xpadded = zeros(TF, Tfull, O, Z)
    copyto!(view(xpadded,1:T,:,:), x)
    dft = rfft(xpadded, 1)
    if σ isa Real
        @tullio r[t,o1,o2] = conj(dft[t,o1,z]) * σ^2 * dft[t,o2,z] threads=false
    else
        @tullio r[t,o1,o2] = conj(dft[t,o1,z]) * σ[z]^2 * dft[t,o2,z] threads=false
    end
    return view(irfft(r, Tfull, 1),1:T,:,:)
end

"""
    allcor!([out], ca::FFTWAllCovCache, x::AbstractArray{<:Real,3}, σ=1)

Same as [`allcov!`](@ref), but transforms results to correlation coefficients
by normalizing the effects of shocks on impact to be one across all observables.
See also [`FFTWAllCovCache`](@ref) and [`allcor`](@ref).

$_allcovargdoc
"""
function allcor!(ca::FFTWAllCovCache, x::AbstractArray{<:Real,3},
        σ::Union{Real, AbstractVector{<:Real}}=1)
    cov = allcov!(ca, x, σ)
    # Avoid allocating a vector for the scales
    @inbounds for j in axes(cov,3)
        sej = sqrt(cov[1,j,j])
        for i in axes(cov,2)
            sei = sqrt(cov[1,i,i])
            v = cov[1,i,j]
            cov[1,i,j] = ifelse(i==j, v, v/(sei*sej))
            for t in 2:size(cov,1)
                cov[t,i,j] /= (sei*sej)
            end
        end
    end
    v = one(eltype(cov))
    @inbounds for k in axes(cov,2)
        cov[1,k,k] = v
    end
    return cov
end

allcor!(out::AbstractArray, ca::FFTWAllCovCache, x::AbstractArray{<:Real,3}, σ=1) =
    copyto!(out, allcor!(ca, x, σ))

"""
    allcor(x::AbstractArray{<:Real,3}, σ::Union{Real, AbstractVector{<:Real}}=1)

Same as [`allcor`](@ref), but transforms results to correlation coefficients
by normalizing the effects of shocks on impact to be one across all observables.
See also [`allcor!`](@ref).

$_allcovargdoc
"""
function allcor(x::AbstractArray{<:Real,3}, σ::Union{Real, AbstractVector{<:Real}}=1)
    cov = allcov(x, σ)
    sd = sqrt.(diag(view(cov,1,:,:)))
    cov .= cov ./ sd' ./ _reshape(sd,1,1,length(sd))
    return cov
end

function _check_correlogram(x, lagmin, lagmax)
    lagmax >= lagmin || throw(ArgumentError("lagmax cannot be smaller than lagmin"))
    Tfull = size(x, 1)
    -Tfull < lagmax < Tfull || throw(ArgumentError("invalid value of lagmax"))
    -Tfull < lagmin < Tfull || throw(ArgumentError("invalid value of lagmin"))
end

"""
    correlogram!(out, ca::FFTWAllCovCache, x::AbstractArray{<:Real,3}, ps; kwargs...)

Same as [`correlogram`](@ref), but avoids memory allocations with `ca`
and store the results to `out`.
See also [`FFTWAllCovCache`](@ref).
"""
function correlogram!(out::AbstractVecOrMat, ca::FFTWAllCovCache,
        x::AbstractArray{<:Real,3}, ps::Union{Pair{Int,Int}, AbstractVector{Pair{Int,Int}}};
        lagmin::Int=0, lagmax::Int=0, σ::Union{Real, AbstractVector{<:Real}}=1)
    _check_correlogram(x, lagmin, lagmax)
    L = lagmax - lagmin + 1
    np = ps isa Pair ? 1 : length(ps)
    (size(out,1), size(out,2)) == (L, np) || throw(ArgumentError(
        "size of out ($(size(out))) does not match the ps or range of lags"))
    cor = allcor!(ca, x, σ)
    rpos = lagmax>=0 ? (max(lagmin,0)+1:lagmax+1) : ()
    npos = length(rpos)
    rneg = lagmin<0 ? (-lagmin+1:-1:-min(lagmax,-1)+1) : ()
    nneg = length(rneg)
    ps isa Pair && (ps = (ps,))
    for (i, p) in enumerate(ps)
        nneg > 0 && copyto!(view(out,1:nneg,i), view(cor,rneg,p[1],p[2]))
        npos > 0 && copyto!(view(out,nneg+1:nneg+npos,i), view(cor,rpos,p[2],p[1]))
    end
    return out
end

"""
    correlogram(x::AbstractArray{<:Real,3}, ps; kwargs...)

Return a correlogram for the vector moving-average process
represented by coefficients in `x`.
See also [`correlogram!`](@ref).

$_allcovargdoc

The observable variables involved in the correlation coefficients
are specified as either a `Pair{Int,Int}` or a vector of `Pair{Int,Int}`s
with the integer index of a variable paired with another.
Correlation coefficients of the first variable relative to the second one in each pair
with different time lags are collected columnwise in the returned matrix.
The range of the time lags are specified with keywords `lagmin` and `lagmax`.

# Keywords
- `lagmin::Int=0`: the smallest time lag to be considered.
- `lagmax::Int=0`: the largest time lag to be considered.
- `σ::Union{Real, AbstractVector{<:Real}}=1`: shock standard deviation(s).
"""
function correlogram(x::AbstractArray{<:Real,3},
        ps::Union{Pair{Int,Int}, AbstractVector{Pair{Int,Int}}};
        lagmin::Int=0, lagmax::Int=0, σ::Union{Real, AbstractVector{<:Real}}=1)
    _check_correlogram(x, lagmin, lagmax)
    TF = promote_type(eltype(x), σ isa Real ? typeof(σ) : eltype(σ))
    L = lagmax - lagmin + 1
    np = ps isa Pair ? 1 : length(ps)
    out = Matrix{TF}(undef, L, np)
    cor = allcor(x, σ)
    rpos = lagmax>=0 ? (max(lagmin,0)+1:lagmax+1) : ()
    npos = length(rpos)
    rneg = lagmin<0 ? (-lagmin+1:-1:-min(lagmax,-1)+1) : ()
    nneg = length(rneg)
    ps isa Pair && (ps = (ps,))
    for (i, p) in enumerate(ps)
        nneg > 0 && copyto!(view(out,1:nneg,i), view(cor,rneg,p[1],p[2]))
        npos > 0 && copyto!(view(out,nneg+1:nneg+npos,i), view(cor,rpos,p[2],p[1]))
    end
    return out
end

function _fill_allcov!(V::AbstractMatrix{TF}, r::AbstractArray{TF,3},
        error::Union{AbstractVector{TF},Nothing}=nothing) where TF<:AbstractFloat
    T, O, _ = size(r)
    N = size(V, 2)
    Tobs = Int(N/O)
    z = zero(TF)
    @inbounds for o2 in 1:O
        for t2 in 1:Tobs
            for o1 in 1:O
                for t1 in 1:Tobs
                    l = abs(t1-t2)+1
                    i = t1+(o1-1)*Tobs
                    j = t2+(o2-1)*Tobs
                    if l > T
                        V[i,j] = z
                    elseif t1 < t2
                        V[i,j] = r[l,o1,o2]
                    elseif t1 > t2
                        V[i,j] = r[l,o2,o1]
                    else
                        V[i,j] = r[1,o1,o2]
                        if error !== nothing && o1 == o2
                            V[i,j] += error[o1]
                        end
                    end
                end
            end
        end
    end
    return V
end

"""
    loglikelihood!(V::AbstractMatrix, Y::AbstractVector, Ycache=similar(Y))

Return the log likelihood of observation `Y` given a multivariate normal distribution
with zero means and variance-covariance matrix `V`.
Elements in `V` are overwritten for intermediate steps;
while `Y` is not mutated.
Memory allocations can be avoided by providing an additional vector `Ycache`.
"""
function loglikelihood!(V::AbstractMatrix, Y::AbstractVector,
        Ycache::AbstractVector=similar(Y))
    N = length(Y)
    # Hermitian avoids some cases where cholesky! fails
    chol = cholesky!(Hermitian(V))
    # A multiple of 2 for the log determinant is cancelled out by the root
    D = sum(log, view(_reshape(V, N^2), diagind(V)))
    # _reshape avoids allocations
    ldiv!(_reshape(Ycache,N,1), chol, _reshape(Y,N,1))
    Q = BLAS.dot(N, Y, 1, Ycache, 1)
    return -D - Q/2
end

