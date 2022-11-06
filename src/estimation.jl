# Assume x is already padded with zeros
function _allcov!(r::AbstractArray{<:Complex,3}, x::AbstractArray{<:Real,3},
        σ::AbstractVector{<:Real})
    N = size(x, 1)
    dft = rfft(x, 1)
    @tullio r[t,o1,o2] = conj(dft[t,o1,z]) * σ[z]^2 * dft[t,o2,z]
    return irfft(r, N, 1)
end

function _allcov!(r::AbstractArray{<:Complex,3}, x::AbstractArray{<:Real,3}, σ::Real)
    N = size(x, 1)
    dft = rfft(x, 1)
    @tullio r[t,o1,o2] = conj(dft[t,o1,z]) * σ^2 * dft[t,o2,z]
    return irfft(r, N, 1)
end

function allcov!(r::AbstractArray{<:Complex{TF},3}, x::AbstractArray{<:Real,3},
        σ::Union{AbstractVector{<:Real},Real}=1) where TF<:AbstractFloat
    T, O, Z = size(x)
    # Pad x with zeros
    x1 = zeros(TF, 2*T-2, O, Z)
    copyto!(view(x1,1:T,:,:), x)
    return view(_allcov!(r, x1, σ),1:T,:,:)
end

function allcov(x::AbstractArray{<:Real,3}, σ::Union{AbstractVector{<:Real},Real}=1)
    T, O, _ = size(x)
    TF = promote_type(eltype(x), σ isa Real ? typeof(σ) : eltype(σ))
    r = Array{Complex{TF},3}(undef, T, O, O)
    return allcov!(r, x, σ)
end

function allcor!(r::AbstractArray{<:Complex,3}, x::AbstractArray{<:Real,3},
        σ::Union{AbstractVector{<:Real},Real}=1)
    cov = allcov!(r, x, σ)
    sd = sqrt.(diag(view(cov,1,:,:)))
    cov .= cov ./ sd' ./ reshape(sd,1,1,length(sd))
    return cov
end

function allcor(x::AbstractArray{<:Real,3}, σ::Union{AbstractVector{<:Real},Real}=1)
    T, O, _ = size(x)
    TF = promote_type(eltype(x), σ isa Real ? typeof(σ) : eltype(σ))
    r = Array{Complex{TF},3}(undef, T, O, O)
    return allcor!(r, x, σ)
end

function correlogram!(out::AbstractVecOrMat, r::AbstractArray{<:Complex,3},
        x::AbstractArray{<:Real,3}, ps::Union{Pair{Int,Int}, AbstractVector{Pair{Int,Int}}};
        lagmin::Int=0, lagmax::Int=0, σ::Union{AbstractVector{<:Real},Real}=1)
    lagmax >= lagmin || throw(ArgumentError("lagmax cannot be smaller than lagmin"))
    Tfull = size(x, 1)
    -Tfull < lagmax < Tfull || throw(ArgumentError("invalid value of lagmax"))
    -Tfull < lagmin < Tfull || throw(ArgumentError("invalid value of lagmin"))
    L = lagmax - lagmin + 1
    np = ps isa Pair ? 1 : length(ps)
    size(out) == (L, np) || throw(ArgumentError(
        "size of out ($(size(out))) does not match the ps or range of lags"))
    cor = allcor!(r, x, σ)
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

function correlogram(x::AbstractArray{<:Real,3},
        ps::Union{Pair{Int,Int}, AbstractVector{Pair{Int,Int}}};
        lagmin::Int=0, lagmax::Int=0, σ::Union{AbstractVector{<:Real},Real}=1)
    TF = promote_type(eltype(x), σ isa Real ? typeof(σ) : eltype(σ))
    out = Matrix{TF}(undef, lagmax-lagmin+1, ps isa Pair ? 1 : length(ps))
    T, O, _ = size(x)
    r = Array{Complex{TF},3}(undef, T, O, O)
    return correlogram!(out, r, x, ps; lagmin=lagmin, lagmax=lagmax, σ=σ)
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

function allcov!(V::AbstractMatrix{TF}, r::AbstractArray{TF,3},
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

function loglikelihood!(Y::Vector, Y1::Vector, V::Matrix)
    chol = cholesky!(V)
    D = sum(log, view(chol.factors, diagind(V)))
    Q = BLAS.dot(N, Y, 1, ldiv!(Y1, chol, Y), 1)
    return -D + Q/2
end

loglikelihood!(Y::Vector, V::Matrix) = loglikelihood!(Y, similar(Y), V)

