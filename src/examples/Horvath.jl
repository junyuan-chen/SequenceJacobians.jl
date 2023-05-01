module Horvath

using ..SequenceJacobians
using LinearAlgebra

struct HorvathPlanner{TF<:AbstractFloat}
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
    Ydiff::Vector{TF}
    goods_mkt::Vector{TF}
    euler::Vector{TF}
    invpiece::Vector{TF}
    multmat::Matrix{TF}
    μY::Vector{TF}
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
    Ydiff = Vector{TF}(undef, N)
    goods_mkt = Vector{TF}(undef, N)
    euler = Vector{TF}(undef, N)
    invpiece = Vector{TF}(undef, N)
    multmat = Matrix{TF}(undef, N, N)
    μY = Vector{TF}(undef, N)
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
    return HorvathPlanner(α, ξ, θ, Γ, Λ, δ, ρA, C, A, K, I, Z, L, X, Y, VA, μ, ζ,
        ζdiff, Ydiff, goods_mkt, euler,
        invpiece, multmat, μY, xcons, ycons, μcons, μexp, ζk,
        μZ, lZ, lμZ, μrhs, μdiff)
end

function solveμ!(p::HorvathPlanner, hwelast, μdiff, μ)
    p.lZ .= log.(p.μZ./μ)
    p.lZ[.~isfinite.(p.lZ)] .= 0
    p.K .= exp.(p.Λ'*p.lZ .+ 1/hwelast*p.Λ'*p.lμZ .- 1/hwelast.*p.invpiece .-
        log.(p.δ) .- 1/hwelast.*log.(p.ζk.*p.δ))
    μdiff .= exp.(p.μexp\(p.μrhs .- p.θ.*p.α.*log.(p.K))) .- μ
end

# Solutions for steady state follow vom Lehn and Winberry (2021)
function ss(p, β, eis, frisch, hwelast, solver)
    fill!(p.A, 1.0)
    p.invpiece .= view(sum(log.(p.Λ.^p.Λ), dims=1), :)
    p.multmat .= p.Γ .* (1.0.-p.θ)' .+ β .* p.Λ .*
        (p.δ .* p.θ .* p.α ./ (1.0 .- β .* (1.0.-p.δ)))'
    p.μY .= (LinearAlgebra.I - p.multmat)\p.ξ
    p.L .= (1.0.-p.α) .* p.θ .* p.μY
    p.L ./= sum(p.L)^(frisch/(1+frisch))
    Ltot = sum(p.L)
    # kcons == ζk
    p.ζk .= β .* p.θ .* p.α .* p.μY ./ (1.0.-β.*(1.0.-p.δ))
    p.xcons .= (1.0.-p.θ) .* p.μY
    p.ycons .= p.L.^((1.0.-p.α).*p.θ) .* p.ζk.^(p.θ.*p.α) .* p.xcons.^(1.0.-p.θ)
    p.μcons .= (view(prod(p.Λ.^p.Λ, dims=1),:)).^(p.θ.*p.α) .*
        (view(prod(p.Γ.^p.Γ, dims=1),:)).^(1.0.-p.θ)
    p.μexp .= LinearAlgebra.I - p.Γ'.*(1.0.-p.θ) - p.Λ'.*(p.θ.*p.α)
    p.μZ .= p.Λ * (p.δ .* p.ζk)
    p.μ .= exp.(p.μexp \ (log.(p.μY) .- log.(p.μcons) .- log.(p.ycons)))
    if hwelast == -1
        p.ζ .= exp.(p.Λ'*log.(p.μ) .- p.invpiece)
        p.Y .= p.μY ./ p.μ
        p.K .= p.ζk ./ p.ζ
    else
        p.lμZ .= log.(p.μZ)
        p.lμZ[.~isfinite.(p.lμZ)] .= 0
        # Must take log after prod as Γ contains 0s
        p.μrhs .= log.(p.μY) .- (1.0.-p.θ).*log.(view(prod(p.Γ.^p.Γ, dims=1),:)) .-
            (1.0.-p.α).*p.θ.*log.(p.L) .- (1.0.-p.θ).*log.(p.xcons)
        p.μexp .= LinearAlgebra.I - p.Γ'.*(1.0.-p.θ)
        r = solve(solver, (f,x)->solveμ!(p,hwelast,f,x), p.μ, ftol=1e-9)
        p.μ .= r.x # Assume solver is from NonlinearSystems.jl
        p.ζ .= p.ζk ./ p.K
        p.Y .= p.μY ./ p.μ
    end
    p.Z .= p.μZ ./ p.μ
    Ctot = prod((p.ξ./p.μ).^p.ξ).^(1/eis)
    p.C .= p.ξ./p.μ.*Ctot.^(1.0.-eis)
    p.I .= p.δ .* p.K
    p.X .= p.μY .* (1.0.-p.θ) .* view(prod((p.Γ./p.μ).^p.Γ, dims=1),:)
    p.VA .= p.K.^p.α .* p.L.^(1.0.-p.α)
    C, K, I, Z, L, X, Y, VA, μ, ζ, A, μY =
        p.C, p.K, p.I, p.Z, p.L, p.X, p.Y, p.VA, p.μ, p.ζ, p.A, p.μY
    return (p=p, Ctot=Ctot, Ltot=Ltot, C=C, K=K, I=I, Z=Z, L=L, X=X, Y=Y, VA=VA, μ=μ, ζ=ζ,
        A=A, μY=μY)
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

# sA1 and sA2 are added here solely for testing purposes
@simple function valueadded(p, A, K, L, sA1, sA2)
    VA = p.VA
    VA .= sA1 .* sA2 .* A.^(1.0./p.θ) .* lag(K).^p.α .* L.^(1.0.-p.α)
    return VA
end

@simple function intermediate(p, μ, Y)
    X = p.X
    N = length(X)
    for j in 1:N
        X[j] = μ[j] * Y[j] * (1.0-p.θ[j])
        for i in 1:N
            γ = p.Γ[i,j]
            X[j] *= (γ/μ[i])^γ
        end
    end
    return X
end

@simple function production(p, VA, X, Y)
    Ydiff = p.Ydiff
    Ydiff .= VA.^p.θ .* X.^(1.0.-p.θ) .- Y
    return Ydiff
end

@simple function investment(p, K)
    I = p.I
    I .= K .- (1.0.-p.δ) .* lag(K)
    return I
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

@simple function goods_mkt(p, Y, C, Z, μ)
    N = length(p.α)
    goods_mkt = p.goods_mkt
    for i in 1:N
        goods_mkt[i] = Y[i] - C[i] - Z[i]
        for j in 1:N
            goods_mkt[i] -= μ[j]/μ[i]*(1-p.θ[j])*p.Γ[i,j]*Y[j]
        end
    end
    return goods_mkt
end

@simple function euler(p, ζ, β, μ, Y, K)
    euler = p.euler
    euler .= β .* (p.α .* p.θ .* lead(μ) .* lead(Y) ./ K .+ lead(ζ) .* (1.0.-p.δ)) .- ζ
    return euler
end

function Horvathmodel(p, calis, solver, sparseH_U)
    N = length(p.α)
    calisbY = [:p=>p, :μ=>p.μ, :A=>p.A, :sA1=>1.0, :sA2=>ones(length(p.A)),
        :K=>p.K, :frisch=>calis[:frisch]]
    calisbζ = [:p=>p, :μ=>p.μ, :I=>p.I, :hwelast=>calis[:hwelast]]
    bY = block([labor_blk(), labortot_blk(), valueadded_blk(), intermediate_blk(),
        production_blk()], [:μ, :A, :K, :sA1, :sA2], [:Y, :Ltot, :L, :VA, :X],
        calisbY, [:Y=>p.Y, :Ltot=>sum(p.L)],
        [:Ydiff=>zeros(N), :Ltotdiff=>0.0], solver=solver, sparseH_U=sparseH_U)
    bζ = block([investuse_blk(), investprice_blk()],
        [:μ, :I], [:ζ, :Z], calisbζ, :ζ=>p.ζ, :ζdiff=>zeros(N), solver=solver)
    m = model([consumption_blk(), constot_blk(), bY, investment_blk(),
        bζ, goods_mkt_blk(), euler_blk()])
    return m
end

end
