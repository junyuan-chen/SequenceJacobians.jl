abstract type AbstractLawOfMotion{TF<:AbstractFloat} end

"""
    grid(p::AbstractLawOfMotion)

Return the grid associated with `p`.
"""
@inline grid(p::AbstractLawOfMotion) = p.g

abstract type DiscreteTimeLawOfMotion{TF} <: AbstractLawOfMotion{TF} end

"""
    ExogProc{TF} <: DiscreteTimeLawOfMotion{TF}

Law of motion in discrete time for an exogenous state based on a stationary Markov process.

# Fields
- `g::Vector{TF}`: the grid points for state values.
- `m::Matrix{TF}`: the transition matrix with rows for the initial states and columns for the destinations.
- `d::Vector{TF}`: the stationary distribution of the process.
"""
struct ExogProc{TF} <: DiscreteTimeLawOfMotion{TF}
    g::Vector{TF}
    m::Matrix{TF}
    d::Vector{TF}
    function ExogProc{TF}(g::Vector{TF}, m::Matrix{TF}, d::Vector{TF}) where TF
        N = length(g)
        size(m) == (N, N) || throw(DimensionMismatch(
            "size of transition matrix m $(size(m)) does not match length of g ($N)"))
        length(d) == N || throw(DimensionMismatch(
            "length of distribution d $(length(d)) does not match length of g ($N)"))
        all(x->x≈1, sum(m, dims=2)) ||
            throw(ArgumentError("sum of each row of transition matrix m must be 1"))
        sum(d) ≈ 1 ||
            throw(ArgumentError("sum of stationary distribution d must be 1"))
        return new{TF}(g, m, d)
    end
end

"""
    rouwenhorstexp(ρ::Real, σ::Real, n::Int)

Construct an instance of [`ExogProc`](@ref) with a Markov chain of `n` states
that approximates the exponential function of a stationary AR(1) process
with parameters `ρ` and `σ` using the Rouwenhorst method.
This method reproduces results from the method `markov_rouwenhorst` in the Python package.
"""
function rouwenhorstexp(ρ::Real, σ::Real, n::Int)
    p  = (1+ρ)/2
    s, m = _rouwenhorst(p, p, 0, 1, n)
    d = gth_solve!(copy(m))
    scale = σ/sqrt(_variance(s, d))
    y = exp.(s.*scale)
    y ./= dot(d, y)
    return ExogProc{eltype(y)}(y, m, d)
end

"""
    backward!(ev::AbstractArray, v::AbstractArray, ps::ExogProc...)

Compute the expected values `ev` over the state space
given the realized values `v` in the next period and
the law of motion of exogenous states `ps`.
"""
backward!(ev::AbstractMatrix, v::AbstractMatrix, p::ExogProc) = mul!(ev, v, p.m')

function backward!(ev::AbstractArray{T,3}, v::AbstractArray{T,3}, p::ExogProc) where T
    @tullio ev[i,j,k] = v[i,j,l] * p.m[k,l]
end

function backward!(ev::AbstractArray{T,3}, v::AbstractArray{T,3}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio ev[i,j,k] = v[i,l,m] * p1.m[j,l] * p2.m[k,m]
end

function backward!(ev::AbstractArray{T,4}, v::AbstractArray{T,4}, p::ExogProc) where T
    @tullio ev[i,j,k,l] = v[i,j,k,m] * p.m[l,m]
end

function backward!(ev::AbstractArray{T,4}, v::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio ev[i,j,k,l] = v[i,j,m,n] * p1.m[k,m] * p2.m[l,n]
end

function backward!(ev::AbstractArray{T,4}, v::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc, p3::ExogProc) where T
    @tullio ev[i,j,k,l] = v[i,m,n,o] * p1.m[j,m] * p2.m[k,n] * p3.m[l,o]
end

"""
    forward!(out::AbstractArray, D::AbstractArray, ps::ExogProc...)

Compute the distribution of agents over the state space in the next period
given the initial distribution `D` and the exogenous law of motion `ps`.
Results are saved in `out`.
"""
forward!(out::AbstractMatrix, D::AbstractMatrix, p::ExogProc) = mul!(out, D, p.m)

function forward!(out::AbstractArray{T,3}, D::AbstractArray{T,3}, p::ExogProc) where T
    @tullio out[i,j,k] = D[i,j,l] * p.m[l,k]
end

function forward!(out::AbstractArray{T,3}, D::AbstractArray{T,3}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio out[i,j,k] = D[i,l,m] * p1.m[l,j] * p2.m[m,k]
end

function forward!(out::AbstractArray{T,4}, D::AbstractArray{T,4}, p::ExogProc) where T
    @tullio out[i,j,k,l] = D[i,j,k,m] * p.m[m,l]
end

function forward!(out::AbstractArray{T,4}, D::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio out[i,j,k,l] = D[i,j,m,n] * p1.m[m,k] * p2.m[n,l]
end

function forward!(out::AbstractArray{T,4}, D::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc, p3::ExogProc) where T
    @tullio out[i,j,k,l] = D[i,m,n,o] * p1.m[m,j] * p2.m[n,k] * p3.m[o,l]
end

"""
    EndoProc{TF,N} <: DiscreteTimeLawOfMotion{TF}

Law of motion in discrete time for an endogenous state based on the corresponding policy.
See also [`update!`](@ref).

# Fields
- `g::Vector{TF}`: the grid points for state values.
- `li::Array{Int,N}`: indices of the lower-end destinations for the transition given initial states.
- `lp::Array{TF,N}`: fractions of the agents that are assigned to the lower-end destinations given initial states.
"""
struct EndoProc{TF,N} <: DiscreteTimeLawOfMotion{TF}
    g::Vector{TF}
    li::Array{Int,N}
    lp::Array{TF,N}
    function EndoProc{TF,N}(g::Vector{TF}, li::Array{Int,N}, lp::Array{TF,N}) where {TF,N}
        size(li) == size(lp) || throw(DimensionMismatch(
            "size of li $(size(li)) must be the same as size of lp $(size(lp))"))
        return new{TF,N}(g, li, lp)
    end
end

"""
    assetproc(amin::Real, amax::Real, n::Int, dims::Int...; pivot::Real=0.25)

Construct an instance of [`EndoProc`](@ref) with a grid of `n` points
between `amin` and `amax` that are equidistant in log.
The size of the discretized state space need to be specified by `dims`
for constructing arrays needed for forward iteration.

Since the grid points may be nonpositive,
a keyword argument `pivot` can be specified for shifting the grid to the positive part
when computing the distances between adjacent grid points.
The implementation is similar to the method `agrid` in the Python package.
"""
function assetproc(amin::Real, amax::Real, n::Int, dims::Int...; pivot::Real=0.25)
    pivot < 0 && throw(ArgumentError("pivot must be nonnegative"))
    aleft = amin + pivot
    if aleft < 0
        aleft = pivot
        pivot += abs(amin)
    end
    agrid = [exp(a)-pivot for a in range(log(aleft), log(amax+pivot), n)]
    # Ensure the left endpoint is exactly amin
    agrid[1] = amin
    TF = eltype(agrid)
    N = length(dims)
    li = Array{Int,N}(undef, dims...)
    lp = Array{TF,N}(undef, dims...)
    return EndoProc{TF,N}(agrid, li, lp)
end

"""
    update!(p::EndoProc, i::Int, a::AbstractArray)

Translate policy `a` corresponding to the `i`th endogenous state
to probabilistic transition on the discretized state values
based on the method from Young (2010) and store the results in `p`.

# Reference
**Young, Eric R.** 2010. "Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell-Smith Algorithm and Non-stochastic Simulations." *Journal of Economic Dynamics and Control* 34 (1): 36-41.
"""
function update!(p::EndoProc, i::Int, a::AbstractArray)
    size(a) == size(p.li) || throw(DimensionMismatch(
        "incompatible policy array of size $(size(a))"))
    dims = (1:i-1..., i+1:ndims(a)...)
    vlis = splitdimsview(p.li, dims)
    vlps = splitdimsview(p.lp, dims)
    vas = splitdimsview(a, dims)
    for (vli, vlp, va) in zip(vlis, vlps, vas)
        interpolate_coord!(vli, vlp, va, p.g)
    end
    return p
end

"""
    forward!(out::AbstractArray, D::AbstractArray, ps::EndoProc...)

Compute the distribution of agents over the state space
given the initial distribution `D` after agents transit to new states
based on the endogenous law of motion `ps`.
Results are saved in `out`.
"""
function forward!(out::AbstractArray, D::AbstractArray, p::EndoProc)
    size(out) == size(D) || throw(DimensionMismatch(
        "size of out $(size(out)) does not match size of D $(size(D))"))
    N = ndims(out)
    Nexog = N - 1
    if Nexog > 0
        dims = (2:N...,)
        vouts = splitdimsview(out, dims)
        vDs = splitdimsview(D, dims)
        vlis = splitdimsview(p.li, dims)
        vlps = splitdimsview(p.lp, dims)
        for (vout, vD, vli, vlp) in zip(vouts, vDs, vlis, vlps)
            forward_endo!(vout, vD, vli, vlp)
        end
    else
        forward_endo!(out, D, p.li, p.lp)
    end
end

function forward!(out::AbstractArray, D::AbstractArray, p1::EndoProc, p2::EndoProc)
    size(out) == size(D) || throw(DimensionMismatch(
        "size of out $(size(out)) does not match size of D $(size(D))"))
    N = ndims(out)
    Nexog = N - 2
    if Nexog > 0
        dims = (3:N...,)
        vouts = splitdimsview(out, dims)
        vDs = splitdimsview(D, dims)
        vli1s = splitdimsview(p1.li, dims)
        vlp1s = splitdimsview(p1.lp, dims)
        vli2s = splitdimsview(p2.li, dims)
        vlp2s = splitdimsview(p2.lp, dims)
        for (vout, vD, vli1, vlp1, vli2, vlp2) in zip(vouts, vDs, vli1s, vlp1s, vli2s, vlp2s)
            forward_endo!(vout, vD, vli1, vlp1, vli2, vlp2)
        end
    else
        forward_endo!(out, D, vli1, vlp1, vli2, vlp2)
    end
end

function forward_endo!(out, D, lis, lps)
    N = length(D)
    fill!(out, zero(eltype(out)))
    @inbounds for i in 1:N
        li = lis[i]
        lp = lps[i]
        d = D[i]
        out[li] += d * lp
        out[li+1] += d * (1.0-lp)
    end
    return out
end

function forward_endo!(out, D, li1s, lp1s, li2s, lp2s)
    N1, N2 = size(out)
    fill!(out, zero(eltype(out)))
    @inbounds for j in 1:N2
        for i in 1:N1
            li1 = li1s[i,j]
            lp1 = lp1s[i,j]
            li2 = li2s[i,j]
            lp2 = lp2s[i,j]
            d = D[i,j]
            out[li1,li2] += d * lp1 * lp2
            out[li1+1,li2] += d * (1.0-lp1) * lp2
            out[li1,li2+1] += d * lp1 * (1.0-lp2)
            out[li1+1,li2+1] += d * (1.0-lp1) * (1.0-lp2)
        end
    end
    return out
end
