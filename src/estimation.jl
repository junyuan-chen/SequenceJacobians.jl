# Assume x is already padded with zeros
function _autocov!(r::AbstractArray{Complex{TF},3}, x::AbstractArray{TF,3},
        σ::AbstractVector{TF}) where TF<:AbstractFloat
    N = size(x, 1)
    dft = rfft(x, 1)
    @tullio r[t,o1,o2] = conj(dft[t,o1,z]) * σ[z]^2 * dft[t,o2,z]
    return irfft(r, N, 1)
end

function autocov!(r::AbstractArray{Complex{TF},3}, x::AbstractArray{TF,3},
        σ::AbstractVector{TF}) where TF<:AbstractFloat
    T, O, Z = size(x)
    # Pad x with zeros
    x1 = zeros(TF, 2*T-2, O, Z)
    copyto!(view(x1,1:T,:,:), x)
    return view(_autocov!(r, x1, σ),1:T,:,:)
end

function autocov(x::AbstractArray{TF,3}, σ::AbstractVector{TF}) where TF<:AbstractFloat
    T, O, _ = size(x)
    r = Array{Complex{TF},3}(undef, T, O, O)
    return autocov!(r, x, σ)
end

function autocor!(r::AbstractArray{Complex{TF},3}, x::AbstractArray{TF,3},
        σ::AbstractVector{TF}) where TF<:AbstractFloat
    cov = autocov!(r, x, σ)
    sd = sqrt.(diag(view(cov,1,:,:)))
    cov .= cov ./ sd' ./ reshape(sd,1,1,length(sd))
    return cov
end

function autocor(x::AbstractArray{TF,3}, σ::AbstractVector{TF}) where TF<:AbstractFloat
    T, O, _ = size(x)
    r = Array{Complex{TF},3}(undef, T, O, O)
    return autocor!(r, x, σ)
end

function autocov!(V::AbstractMatrix{TF}, r::AbstractArray{TF,3},
        error::Union{Diagonal,UniformScaling,Nothing}=nothing) where TF<:AbstractFloat
    T, O, _ = size(r)
    N = size(V, 2)
    Tobs = Int(N/O)
    V1 = reshape(V, Tobs, O, Tobs, O)
    @inbounds for t2 in 1:Tobs
        for t1 in 1:Tobs
            l = abs(t1-t2)+1
            if l > T
                fill!(view(V1,t1,:,t2,:), zero(TF))
            elseif t1 < t2
                copyto!(view(V1,t1,:,t2,:), view(r,l,:,:))
            elseif t1 > t2
                # Asymmetric
                copyto!(view(V1,t1,:,t2,:), view(r,l,:,:)')
            elseif t1 == t2
                copyto!(view(V1,t1,:,t2,:), view(r,l,:,:))
                error === nothing || (V1[t1,:,t2,:] += error)
            end
        end
    end
    return V
end

function _simul_shock!(dY::AbstractVecOrMat, dX::AbstractVecOrMat, ε::AbstractVecOrMat, nT::Int)
    T = min(length(dY), size(ε,1)-nT+1)
    Nout = size(dY, 2)
    for n in 1:Nout
        for t in 1:T
            # The order of ε is flipped
            dY[t,n] = dot(view(dX,1+(n-1)*nT:n*nT,:), view(ε,t+nT-1:-1:t,:))
        end
    end
    return dY
end

_addssval!(out::VecOrMat, v::Real, T::Int) = (out .+= v)

function _addssval!(out::VecOrMat, vs::AbstractArray, T::Int)
    for (n, v) in enumerate(vs)
        for t in 1:T
            out[t,n] += v
        end
    end
end

function simulate!(out::AbstractVecOrMat, GJ::GEJacobian, exovar::Symbol, endovar::Symbol,
        ε::AbstractVector, ρ::Real, σ::Real=1.0; addssval::Bool=true)
    nT = GJ.nTfull
    G = getM!(GJ, exovar, endovar)
    dX = G * (σ .* ρ.^(0:nT-1))
    _simul_shock!(out, dX, ε, nT)
    if addssval
        T = min(length(out), size(ε,1)-nT+1)
        _addssval!(out, GJ.tjac.varvals[endovar], T)
    end
    return out
end

function simulate!(out::AbstractVecOrMat, GJ::GEJacobian, exovar::Symbol, endovar::Symbol,
        ε::AbstractMatrix, ρ::AbstractVector{<:Real},
        σ::Union{AbstractVector{<:Real},Real}; addssval::Bool=true)
    nT = GJ.nTfull
    G = getM!(GJ, exovar, endovar)
    N = size(ε, 2)
    dX = Matrix{eltype(G)}(undef, size(G,1), N)
    for n in 1:N
        Gn = view(G,:,1+(n-1)*nT:n*nT)
        σn = σ isa Real ? σ : σ[n]
        mul!(view(dX,:,n), Gn, (σn .* ρ[n].^(0:nT-1)))
    end
    _simul_shock!(out, dX, ε, nT)
    if addssval
        T = min(length(out), size(ε,1)-nT+1)
        _addssval!(out, GJ.tjac.varvals[endovar], T)
    end
    return out
end

function simulate(GJ::GEJacobian{T1}, exovar::Symbol, endovar::Symbol,
        ε::AbstractVecOrMat{T2}, ρ, σ=1.0; kwargs...) where {T1,T2}
    G = getG!(GJ, exovar, endovar)
    nout = G isa Matrix || G isa MatMulMap ? size(G,1) : 1
    T = size(ε,1) - GJ.nTfull + 1
    T < 1 && throw(ArgumentError("length of shocks is smaller than $(GJ.nTfull)"))
    out = Matrix{promote_type(T1,T2)}(undef, T, nout)
    simulate!(out, GJ, exovar, endovar, ε, ρ, σ; kwargs...)
    return out
end

function loglikelihood!(Y::Vector, Y1::Vector, V::Matrix)
    chol = cholesky!(V)
    D = sum(log, view(chol.factors, diagind(V)))
    Q = BLAS.dot(N, Y, 1, ldiv!(Y1, chol, Y), 1)
    return -D + Q/2
end

loglikelihood!(Y::Vector, V::Matrix) = loglikelihood!(Y, similar(Y), V)

