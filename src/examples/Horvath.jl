module Horvath

using ..SequenceJacobians
using LinearAlgebra

struct HorvathPlanner{TF<:AbstractFloat,CA}
    α::Vector{TF}
    ξ::Vector{TF}
    θ::Vector{TF}
    Γ::Matrix{TF}
    Λ::Matrix{TF}
    δ::Vector{TF}
    ρA::Vector{TF}
    C::Vector{TF}
    A::Vector{TF}
    K::Vector{TF}
    I::Vector{TF}
    Z::Vector{TF}
    L::Vector{TF}
    X::Vector{TF}
    Y::Vector{TF}
    VA::Vector{TF}
    μ::Vector{TF}
    ζ::Vector{TF}
    ζdiff::Vector{TF}
    goods_mkt::Vector{TF}
    euler::Vector{TF}
    invpiece::Vector{TF}
    multmat::Matrix{TF}
    μy::Vector{TF}
    lcons::Vector{TF}
    kcons::Vector{TF}
    xcons::Vector{TF}
    ycons::Vector{TF}
    μcons::Vector{TF}
    μexp::Matrix{TF}
    ζk::Vector{TF}
    μZ::Vector{TF}
    lZ::Vector{TF}
    lμZ::Vector{TF}
    μrhs::Vector{TF}
    μdiff::Vector{TF}
    ca::CA
end

function HorvathPlanner(para::AbstractDict)
    N = length(para[:vash])
    α = para[:ksh]
    ξ = para[:csh]
    θ = para[:vash]
    Γ = reshape(para[:iomat], N, N)
    Λ = reshape(para[:invmat], N, N)
    δ = para[:δ]
    ρA = para[:ρA]
    TF = Float64
    C = Vector{TF}(undef, N)
    A = Vector{TF}(undef, N)
    K = Vector{TF}(undef, N)
    I = Vector{TF}(undef, N)
    Z = Vector{TF}(undef, N)
    L = Vector{TF}(undef, N)
    X = Vector{TF}(undef, N)
    Y = Vector{TF}(undef, N)
    VA = Vector{TF}(undef, N)
    μ = Vector{TF}(undef, N)
    ζ = Vector{TF}(undef, N)
    ζdiff = Vector{TF}(undef, N)
    goods_mkt = Vector{TF}(undef, N)
    euler = Vector{TF}(undef, N)
    invpiece = Vector{TF}(undef, N)
    multmat = Matrix{TF}(undef, N, N)
    μy = Vector{TF}(undef, N)
    lcons = Vector{TF}(undef, N)
    kcons = Vector{TF}(undef, N)
    xcons = Vector{TF}(undef, N)
    ycons = Vector{TF}(undef, N)
    μcons = Vector{TF}(undef, N)
    μexp = Matrix{TF}(undef, N, N)
    ζk = Vector{TF}(undef, N)
    μZ = Vector{TF}(undef, N)
    lZ = Vector{TF}(undef, N)
    lμZ = Vector{TF}(undef, N)
    μrhs = Vector{TF}(undef, N)
    μdiff = Vector{TF}(undef, N)
    ca = GSL_MultirootFSolverCache(GSL_Hybrids, (f,x)->f, N)
    return HorvathPlanner(α, ξ, θ, Γ, Λ, δ, ρA, C, A, K, I, Z, L, X, Y, VA, μ, ζ,
        ζdiff, goods_mkt, euler,
        invpiece, multmat, μy, lcons, kcons, xcons, ycons, μcons, μexp, ζk,
        μZ, lZ, lμZ, μrhs, μdiff, ca)
end

function solveμ!(p::HorvathPlanner, hwelast, μdiff, μ)
    p.lZ .= log.(p.μZ./μ)
    p.lZ[.~isfinite.(p.lZ)] .= 0
    p.K .= exp.(p.Λ'*p.lZ .+ 1/hwelast*p.Λ'*p.lμZ .- 1/hwelast.*p.invpiece .-
        log.(p.δ) .- 1/hwelast.*log.(p.ζk.*p.δ))
    μdiff .= exp.(p.μexp\(p.μrhs .- (p.θ.*p.α).*log.(p.K))) .- μ
end

# Solutions for steady state follow vom Lehn and Winberry (2021)
@simple function ss(p, β, eis, frisch, hwelast)
    N = length(p.α)
    fill!(p.A, 1.0)
    p.invpiece .= reshape(sum(log.(p.Λ.^p.Λ), dims=1), N)
    p.multmat .= p.Γ .* (1.0.-p.θ)' .+ β .* p.Λ .*
        (p.δ .* p.θ .* p.α ./ (1.0 .- β .* (1.0.-p.δ)))'
    p.μy .= (LinearAlgebra.I - p.multmat)\p.ξ
    p.lcons .= (1.0.-p.α) .* p.θ .* p.μy
    p.lcons ./= sum(p.lcons).^(frisch/(1+frisch))
    p.kcons .= β ./ (1.0.-β.*(1.0.-p.δ)) .* p.α .* p.θ .* p.μy
    p.xcons .= (1.0.-p.θ) .* p.μy
    p.ycons .= p.lcons.^((1.0.-p.α).*p.θ) .* p.kcons.^(p.θ.*p.α) .* p.xcons.^(1.0.-p.θ)
    p.μcons .= (prod(p.Λ.^p.Λ, dims=1)').^(p.θ.*p.α) .*(prod(p.Γ.^p.Γ, dims=1)').^(1.0.-p.θ)
    p.μexp .= LinearAlgebra.I - p.Γ'.*(1.0.-p.θ) - p.Λ'.*(p.θ.*p.α)
    p.ζk .= β .* p.θ .* p.α .* p.μy ./ (1.0.-β.*(1.0.-p.δ))
    p.μZ .= p.Λ * (p.δ .* p.ζk)
    p.μ .= exp.(p.μexp \ (log.(p.μy) .- log.(p.μcons) .- log.(p.ycons)))
    if hwelast == -1
        p.ζ .= exp.(p.Λ'*log.(p.μ) .- p.invpiece)
        p.Y .= p.μy ./ p.μ
        p.K .= β ./ (1.0.-β.*(1.0.-p.δ)) .* (p.θ.*p.α) .* p.μ ./ p.ζ .* p.Y
    else
        p.lμZ .= log.(p.μZ)
        p.lμZ[.~isfinite.(p.lμZ)] .= 0
        # Must take log after prod as Γ contains 0s
        p.μrhs .= log.(p.μy) .- (1.0.-p.θ).*log.(prod(p.Γ.^p.Γ, dims=1)') .-
            ((1.0.-p.α).*p.θ).*log.(p.lcons) .- (1.0.-p.θ).*log.(p.xcons)
        p.μexp .= LinearAlgebra.I - p.Γ'.*(1.0.-p.θ)
        r = solve!(p.ca, (f,x)->solveμ!(p,hwelast,f,x), p.μ, ftol=1e-9)
        p.μ .= r[1]
        p.ζ .= p.ζk ./ p.K
        p.Y .= p.μy ./ p.μ
    end
    p.Z .= p.μZ ./ p.μ
    Ctot = prod((p.ξ./p.μ).^p.ξ).^(1/eis)
    p.C .= p.ξ./p.μ.*Ctot.^(1.0.-eis)
    Lsum = sum((1.0.-p.α) .* p.θ .* p.μy)^(frisch/(1+frisch))
    p.L .= p.μ .* (1.0.-p.α) .* p.θ .* p.Y ./ Lsum
    Ltot = sum(p.L)
    p.I .= p.δ .* p.K
    p.X .= p.μy .* (1.0.-p.θ) .* prod((p.Γ./p.μ).^p.Γ, dims=1)'
    p.VA .= p.K.^p.α .* p.L.^(1.0.-p.α)
    C, K, I, Z, L, X, Y, VA, μ, ζ, A = p.C, p.K, p.I, p.Z, p.L, p.X, p.Y, p.VA, p.μ, p.ζ, p.A
    return Ctot, Ltot, C, K, I, Z, L, X, Y, VA, μ, ζ, A
end

@simple function consumption(p, μ, eis)
    C = p.C
    C .= (μ ./ p.ξ).^(-1.0./eis)
    return C
end

@simple function constot(p, C)
    Ctot = 1.0
    for j in 1:length(p.ξ)
        Ctot *= C[j]^p.ξ[j]
    end
    return Ctot
end

@simple function labor(p, Ltot, μ, Y, frisch)
    L = p.L
    L .= μ .* Y .* (1.0.-p.α) .* p.θ ./ Ltot.^frisch
    return L
end

@simple function labortot(L, Ltot)
    Ltotdiff = sum(L) - Ltot
    return Ltotdiff
end

# All the K is lag K in Dynare
# Need this change to get the solutions
@simple function investment(p, K)
    I = p.I
    I .= lead(K) .- (1.0.-p.δ) .* K
    return I
end

@simple function capital(p, A, Y, L, X)
    K = p.K
    K .= (Y ./ (A .* L.^((1.0.-p.α).*p.θ) .* X.^(1.0.-p.θ))).^(1.0./(p.α.*p.θ))
    return K
end

@simple function intermediate(p, Y, μ)
    X = p.X
    N = length(X)
    for j in 1:N
        X[j] = Y[j] * μ[j] * (1.0-p.θ[j])
        for i in 1:N
            γ = p.Γ[i,j]
            X[j] *= (γ/μ[i])^γ
        end
    end
    return X
end

@simple function investuse(p, ζ, μ, I)
    N = length(p.α)
    Z = p.Z
    for j in 1:N
        Z[j] = 0
        for i in 1:N
            Z[j] += ζ[i] / μ[j] * p.Λ[j,i] * I[i]
        end
    end
    return Z
end

@simple function investprice(p, ζ, μ, I, Z, hwelast)
    N = length(p.α)
    ζdiff = p.ζdiff
    for j in 1:N
        ζdiff[j] = 1
        for i in 1:N
            λ = p.Λ[i,j]
            ζdiff[j] *= (μ[i]/λ)^λ * Z[i]^(λ*(1+hwelast))
        end
        ζdiff[j] = ζdiff[j]/I[j]^(1+hwelast) - ζ[j]
    end
    return ζdiff
end

@simple function goods_mkt(p, Y, C, μ, ζ, I)
    N = length(p.α)
    goods_mkt = p.goods_mkt
    for i in 1:N
        goods_mkt[i] = Y[i] - C[i]
        for j in 1:N
            goods_mkt[i] = goods_mkt[i] - μ[j]/μ[i]*(1-p.θ[j])*p.Γ[i,j]*Y[j] -
                ζ[j]/μ[i]*p.Λ[i,j]*I[j]
        end
    end
    return goods_mkt
end

@simple function euler(p, ζ, β, μ, Y, K)
    euler = p.euler
    euler .= β .* (μ .* p.α .* p.θ .* Y ./ K .+ ζ .* (1.0.-p.δ)) .- lag(ζ)
    return euler
end

function Horvathmodel(p, calis)
    push!(calis, :μ=>p.μ, :I=>p.I, :Y=>p.Y)
    bζ = block([investuse_blk(), investprice_blk()],
        [:μ, :I], [:ζ, :Z], calis, :ζ=>p.ζ, :ζdiff=>zeros(length(p.ζ)),
        solver=GSL_Hybrids, static=true)
    bL = block([labor_blk(), labortot_blk()],
        [:μ, :Y], [:Ltot, :L], calis, :Ltot=>sum(p.L), :Ltotdiff=>0.0,
        solver=GSL_Hybrids, static=true)
    m = model([consumption_blk(), constot_blk(), bL, investment_blk(), capital_blk(),
        intermediate_blk(), bζ, goods_mkt_blk(), euler_blk()])
    return m
end

end
