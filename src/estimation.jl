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
                # Minor deviation from symmetry might exist
                copyto!(view(V1,t1,:,t2,:), view(r,l,:,:)')
            elseif t1 == t2
                copyto!(view(V1,t1,:,t2,:), view(r,l,:,:))
                error === nothing || (V1[t1,:,t2,:] += error)
            end
        end
    end
    return V
end

function loglikelihood!(Y::Vector, Y1::Vector, V::Matrix)
    chol = cholesky!(V)
    D = sum(log, view(chol.factors, diagind(V)))
    Q = BLAS.dot(N, Y, 1, ldiv!(Y1, chol, Y), 1)
    return -D + Q/2
end

loglikelihood!(Y::Vector, V::Matrix) = loglikelihood!(Y, similar(Y), V)

