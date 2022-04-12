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
the exogenous law of motion `ps`.
"""
@inline backward!(ev::AbstractMatrix, v::AbstractMatrix, p::ExogProc) = mul!(ev, v, p.m')

@inline function backward!(ev::AbstractArray{T,3}, v::AbstractArray{T,3}, p::ExogProc) where T
    @tullio ev[i,j,k] = v[i,j,l] * p.m[k,l]
end

@inline function backward!(ev::AbstractArray{T,3}, v::AbstractArray{T,3}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio ev[i,j,k] = v[i,l,m] * p1.m[j,l] * p2.m[k,m]
end

@inline function backward!(ev::AbstractArray{T,4}, v::AbstractArray{T,4}, p::ExogProc) where T
    @tullio ev[i,j,k,l] = v[i,j,k,m] * p.m[l,m]
end

@inline function backward!(ev::AbstractArray{T,4}, v::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio ev[i,j,k,l] = v[i,j,m,n] * p1.m[k,m] * p2.m[l,n]
end

@inline function backward!(ev::AbstractArray{T,4}, v::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc, p3::ExogProc) where T
    @tullio ev[i,j,k,l] = v[i,m,n,o] * p1.m[j,m] * p2.m[k,n] * p3.m[l,o]
end

"""
    forward!(out::AbstractArray, D::AbstractArray, ps::ExogProc...)

Compute the distribution of agents over the state space in the next period
given the initial distribution `D` and the exogenous law of motion `ps`.
Results are saved in `out`.
"""
@inline forward!(out::AbstractMatrix, D::AbstractMatrix, p::ExogProc) = mul!(out, D, p.m)

@inline function forward!(out::AbstractArray{T,3}, D::AbstractArray{T,3}, p::ExogProc) where T
    @tullio out[i,j,k] = D[i,j,l] * p.m[l,k]
end

@inline function forward!(out::AbstractArray{T,3}, D::AbstractArray{T,3}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio out[i,j,k] = D[i,l,m] * p1.m[l,j] * p2.m[m,k]
end

@inline function forward!(out::AbstractArray{T,4}, D::AbstractArray{T,4}, p::ExogProc) where T
    @tullio out[i,j,k,l] = D[i,j,k,m] * p.m[m,l]
end

@inline function forward!(out::AbstractArray{T,4}, D::AbstractArray{T,4}, p1::ExogProc,
        p2::ExogProc) where T
    @tullio out[i,j,k,l] = D[i,j,m,n] * p1.m[m,k] * p2.m[n,l]
end

@inline function forward!(out::AbstractArray{T,4}, D::AbstractArray{T,4}, p1::ExogProc,
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
    assetgrid(amin::Real, amax::Real, n::Int; pivot::Real=0.25)

Return a grid of `n` points between `amin` and `amax` that are equidistant in log.
The implementation is similar to the method `agrid` in the Python package.
See also [`assetproc`](@ref) that additionally supports
transitions across state space.
"""
function assetgrid(amin::Real, amax::Real, n::Int; pivot::Real=0.25)
    pivot < 0 && throw(ArgumentError("pivot must be nonnegative"))
    aleft = amin + pivot
    if aleft < 0
        aleft = pivot
        pivot += abs(amin)
    end
    agrid = [exp(a)-pivot for a in range(log(aleft), log(amax+pivot), n)]
    # Ensure the left endpoint is exactly amin
    agrid[1] = amin
    return agrid
end

"""
    assetproc(amin::Real, amax::Real, n::Int, dims::Int...; pivot::Real=0.25)

Construct an instance of [`EndoProc`](@ref) with a grid of `n` points
between `amin` and `amax` that are equidistant in log.
The size of the discretized state space need to be specified by `dims`
for constructing arrays needed for forward iteration.
See also [`assetgrid`](@ref), which only returns the grid.

Since the grid points may be nonpositive,
a keyword argument `pivot` can be specified for shifting the grid to the positive part
when computing the distances between adjacent grid points.
The implementation is similar to the method `agrid` in the Python package.
"""
function assetproc(amin::Real, amax::Real, n::Int, dims::Int...; pivot::Real=0.25)
    agrid = assetgrid(amin, amax, n; pivot=pivot)
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
    @inbounds for k in eachindex(vlis)
        interpolate_coord!(vlis[k], vlps[k], vas[k], p.g)
    end
    return p
end

"""
    backward!(ev::AbstractArray, v::AbstractArray, ps::EndoProc...)

Compute the expected values `ev` over the state space
given the realized values `v` in the next period and
the endogenous law of motion `ps`.
"""
function backward!(ev::AbstractArray, v::AbstractArray, p::EndoProc)
    size(ev) == size(v) || throw(DimensionMismatch(
        "size of ev $(size(ev)) does not match size of v $(size(v))"))
    N = ndims(ev)
    Nexog = N - 1
    if Nexog > 0
        dims = (2:N...,)
        vevs = splitdimsview(ev, dims)
        vvs = splitdimsview(v, dims)
        vlis = splitdimsview(p.li, dims)
        vlps = splitdimsview(p.lp, dims)
        @inbounds for k in eachindex(vevs)
            backward_endo!(vevs[k], vvs[k], vlis[k], vlps[k])
        end
    else
        backward_endo!(ev, v, p.li, p.lp)
    end
end

function backward!(ev::AbstractArray, v::AbstractArray, p1::EndoProc, p2::EndoProc)
    size(ev) == size(v) || throw(DimensionMismatch(
        "size of ev $(size(ev)) does not match size of v $(size(v))"))
    N = ndims(ev)
    Nexog = N - 2
    if Nexog > 0
        dims = (3:N...,)
        vevs = splitdimsview(ev, dims)
        vvs = splitdimsview(v, dims)
        vli1s = splitdimsview(p1.li, dims)
        vlp1s = splitdimsview(p1.lp, dims)
        vli2s = splitdimsview(p2.li, dims)
        vlp2s = splitdimsview(p2.lp, dims)
        @inbounds for k in eachindex(vevs)
            backward_endo!(vevs[k], vvs[k], vli1s[k], vlp1s[k], vli2s[k], vlp2s[k])
        end
    else
        backward_endo!(ev, v, p1.li, p1.lp, p2.li, p2.lp)
    end
end

function backward_endo!(ev::AbstractVector, v::AbstractVector,
        lis::AbstractVector, lps::AbstractVector)
    N = length(ev)
    fill!(ev, zero(eltype(ev)))
    @inbounds for i in 1:N
        li = lis[i]
        lp = lps[i]
        ev[i] = lp * v[li] + (1.0-lp) * v[li+1]
    end
    return ev
end

function backward_endo!(ev::AbstractMatrix, v::AbstractMatrix,
        li1s::AbstractMatrix, lp1s::AbstractMatrix, li2s::AbstractMatrix, lp2s::AbstractMatrix)
    N1, N2 = size(ev)
    fill!(ev, zero(eltype(ev)))
    @inbounds for j in 1:N2
        for i in 1:N1
            li1 = li1s[i,j]
            lp1 = lp1s[i,j]
            li2 = li2s[i,j]
            lp2 = lp2s[i,j]
            ev[i,j] = lp1 * lp2 * v[li1,li2] + lp1 * (1.0-lp2) * v[li1,li2+1] +
                (1.0-lp1) * lp2 * v[li1+1,li2] + (1.0-lp1) * (1.0-lp2) * v[li1+1,li2+1]
        end
    end
    return ev
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
        @inbounds for k in eachindex(vouts)
            forward_endo!(vouts[k], vDs[k], vlis[k], vlps[k])
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
        @inbounds for k in eachindex(vouts)
            forward_endo!(vouts[k], vDs[k], vli1s[k], vlp1s[k], vli2s[k], vlp2s[k])
        end
    else
        forward_endo!(out, D, p1.li, p1.lp, p2.li, p2.lp)
    end
end

function forward_endo!(out::AbstractVector, D::AbstractVector,
        lis::AbstractVector, lps::AbstractVector)
    N = length(out)
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

function forward_endo!(out::AbstractMatrix, D::AbstractMatrix,
        li1s::AbstractMatrix, lp1s::AbstractMatrix, li2s::AbstractMatrix, lp2s::AbstractMatrix)
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

# Compute curly Ds
function forward_shock!(out::AbstractArray, D::AbstractArray, p::EndoProc, da::AbstractArray)
    size(out) == size(D) == size(da) || throw(DimensionMismatch(
        "size of out $(size(out)), D $(size(D)) and da $(size(da)) must be the same"))
    N = ndims(out)
    Nexog = N - 1
    if Nexog > 0
        dims = (2:N...,)
        vouts = splitdimsview(out, dims)
        vDs = splitdimsview(D, dims)
        vlis = splitdimsview(p.li, dims)
        vdas = splitdimsview(da, dims)
        @inbounds for k in eachindex(vouts)
            forward_shock_endo!(vouts[k], vDs[k], vlis[k], vdas[k], p.g)
        end
    else
        forward_shock_endo!(out, D, p.li, da, p.g)
    end
end

function forward_shock!(out::AbstractArray, D::AbstractArray,
        p1::EndoProc, p2::EndoProc, da1::AbstractArray, da2::AbstractArray)
    size(out) == size(D) == size(da1) == size(da2) || throw(DimensionMismatch(
        "size of out $(size(out)), D $(size(D)), da1 $(size(da1)) and da2 $(size(da2)) must be the same"))
    N = ndims(out)
    Nexog = N - 2
    if Nexog > 0
        dims = (3:N...,)
        vouts = splitdimsview(out, dims)
        vDs = splitdimsview(D, dims)
        vli1s = splitdimsview(p1.li, dims)
        vlp1s = splitdimsview(p1.lp, dims)
        vda1s = splitdimsview(da1, dims)
        vli2s = splitdimsview(p2.li, dims)
        vlp2s = splitdimsview(p2.lp, dims)
        vda2s = splitdimsview(da2, dims)
        @inbounds for k in eachindex(vouts)
            forward_shock_endo!(vouts[k], vDs[k], vli1s[k], vlp1s[k], vda1s[k],
                vli2s[k], vlp2s[k], vda2s[k], p1.g, p2.g)
        end
    else
        forward_shock_endo!(out, D, p1.li, p1.lp, da1, p2.li, p2.lp, da2, p1.g, p2.g)
    end
end

function forward_shock_endo!(out::AbstractVector, D::AbstractVector,
        lis::AbstractVector, da::AbstractVector, g::AbstractVector)
    N = length(out)
    fill!(out, zero(eltype(out)))
    @inbounds for i in 1:N
        li = lis[i]
        d = da[i] / (g[li+1] - g[li]) * D[i]
        out[li] -= d
        out[li+1] += d
    end
    return out
end

function forward_shock_endo!(out::AbstractMatrix, D::AbstractMatrix,
        li1s::AbstractMatrix, lp1s::AbstractMatrix, da1::AbstractMatrix,
        li2s::AbstractMatrix, lp2s::AbstractMatrix, da2::AbstractMatrix,
        g1::AbstractVector, g2::AbstractVector)
    N1, N2 = size(out)
    fill!(out, zero(eltype(out)))
    @inbounds for j in 1:N2
        for i in 1:N1
            li1 = li1s[i,j]
            lp1 = lp1s[i,j]
            li2 = li2s[i,j]
            lp2 = lp2s[i,j]
            d = D[i,j]
            d1 = da1[i,j] / (g1[li1+1] - g1[li1]) * d
            d2 = da2[i,j] / (g2[li2+1] - g2[li2]) * d
            out[li1,li2] = out[li1,li2] - d1 * lp2 - lp1 * d2
            out[li1+1,li2] = out[li1+1,li2] + d1 * lp2 - (1.0-lp1) * d2
            out[li1,li2+1] = out[li1,li2+1] - d1 * (1.0-lp2) + lp1 * d2
            out[li1+1,li2+1] = out[li1+1,li2+1] + d1 * (1.0-lp2) + (1.0-lp1) * d2
        end
    end
    return out
end
