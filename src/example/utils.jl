module ExampleUtils

export grida, gridrouwenhorst, forward_policy_1d!

"""
    grida(amax::Real, n::Int, amin::Real=0)

Create a grid with `n` points between `amin` and `amax` that are equidistant in log.
This method reproduces results from the method `agrid` in the Python package.
"""
function grida(amax::Real, n::Int, amin::Real=0.0)
    pivot = abs(amin) + 0.25
    agrid = [exp(a)-pivot for a in range(log(amin+pivot), log(amax+pivot), n)]
    agrid[1] = amin
    return agrid
end

variance(x::AbstractArray, pr::AbstractArray) = sum(pr.*(x.-sum(pr.*x)).^2)


"""
    gridrouwenhorst(ρ::Real, σ::Real, n::Int)

Create a grid with `n` points that approximates a stationary AR(1) process
with parameters `ρ` and `σ` using the Rouwenhorst method.
This method reproduces results from the method `markov_rouwenhorst` in the Python package.
"""
function gridrouwenhorst(ρ::Real, σ::Real, n::Int)
    p  = (1+ρ)/2
    s, Pi = _rouwenhorst(p, p, 0, 1, n)
    pr = gth_solve!(copy(Pi))
    scale = σ/sqrt(variance(s, pr))
    y = exp.(s.*scale)
    y ./= sum(pr.*y)
    return y, pr, Pi
end

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

function _getk(knots, xi, k0, K)
    # Start the search from the last result
    k = searchsortedfirst(view(knots, k0+1:K), xi)
    knext = k0+k-1
    if k0+k > 1
        k -= 1
        if k0+k == K
            k -= 1
        end
    end
    return k, knext
end

function forward_policy_1d!(Dnew, D, xi, xpi)
    nX, nZ = size(D)
    fill!(Dnew, zero(eltype(Dnew)))
    for iz in 1:nZ
        for ix in 1:nX
            i = xi[ix,iz]
            pr = xpi[ix,iz]
            d = D[ix,iz]
            Dnew[i,iz] += d * pr
            Dnew[i+1,iz] += d * (1-pr)
        end
    end
    return Dnew
end

end
