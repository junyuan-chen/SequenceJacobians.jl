module TwoAsset

using ..SequenceJacobians
using ..SequenceJacobians.ExampleUtils
using LinearAlgebra: mul!
using SplitApplyCombine: splitdimsview

import SequenceJacobians: endostates, endopolicies, exogstates, valuevars, policies,
    backwardtargets, backward_init!, backward_endo!

export twoassethhblock, twoassetmodelss, twoassetmodel

struct TwoAssetHousehold{TF<:AbstractFloat} <: AbstractHetAgent
    aproc::EndoProc{TF,3}
    bproc::EndoProc{TF,3}
    eproc::ExogProc{TF}
    κgrid::Vector{TF}
    zgrid::Vector{TF}
    li0::Array{Int,3}
    lp0::Array{TF,3}
    a_endo_unc::Array{TF,3}
    c_endo_unc::Array{TF,3}
    b_endo::Array{TF,3}
    a::Array{TF,3}
    alast::Array{TF,3}
    b::Array{TF,3}
    blast::Array{TF,3}
    lhs_con::Array{TF,3}
    li1::Array{Int,3}
    lp1::Array{TF,3}
    a_endo_con::Array{TF,3}
    c_endo_con::Array{TF,3}
    a_con::Array{TF,3}
    Ψ1grid::Matrix{TF}
    achange::Array{TF,3}
    corefactor::Array{TF,3}
    Ψ::Array{TF,3}
    Ψ1::Array{TF,3}
    Ψ2::Array{TF,3}
    c::Array{TF,3}
    uc::Array{TF,3}
    uce::Array{TF,3}
    Va::Array{TF,3}
    Vb::Array{TF,3}
    EVa::Array{TF,3}
    EVb::Array{TF,3}
    Wb::Array{TF,3}
    Wratio::Array{TF,3}
    D::Array{TF,3}
    Dendo::Array{TF,3}
    Dlast::Array{TF,3}
end

function TwoAssetHousehold(amax, bmax, κmax, Na, Nb, Ne, Nκ, ρe, σe)
    dims0 = (Na, Nb, Ne)
    dims1 = (Na, Nκ, Ne)
    aproc = assetproc(0, amax, Na, dims0...)
    bproc = assetproc(0, bmax, Nb, dims0...)
    eproc = rouwenhorstexp(ρe, σe, Ne)
    κgrid = reverse!(assetgrid(0, κmax, Nκ))
    zgrid = similar(grid(eproc))
    li0 = Array{Int,3}(undef, dims0)
    lp0 = Array{Float64,3}(undef, dims0)
    a_endo_unc = similar(lp0)
    c_endo_unc = similar(lp0)
    b_endo = similar(lp0)
    a = similar(lp0)
    alast = similar(lp0)
    b = similar(lp0)
    blast = similar(lp0)
    lhs_con = Array{Float64,3}(undef, dims1)
    li1 = Array{Int,3}(undef, dims1)
    lp1 = Array{Float64,3}(undef, dims1)
    a_endo_con = similar(lp1)
    c_endo_con = similar(lp1)
    a_con = similar(lp1)
    Ψ1grid = Matrix{Float64}(undef, Na, Na)
    achange = similar(lp0)
    corefactor = similar(lp0)
    Ψ = similar(lp0)
    Ψ1 = similar(lp0)
    Ψ2 = similar(lp0)
    c = similar(lp0)
    uc = similar(lp0)
    uce = similar(lp0)
    Va = similar(lp0)
    Vb = similar(lp0)
    EVa = similar(lp0)
    EVb = similar(lp0)
    Wb = similar(lp0)
    Wratio = similar(lp0)
    D = similar(lp0)
    Dendo = similar(lp0)
    Dlast = similar(lp0)
    return TwoAssetHousehold{eltype(c)}(aproc, bproc, eproc, κgrid, zgrid, li0, lp0,
        a_endo_unc, c_endo_unc, b_endo, a, alast, b, blast, lhs_con, li1, lp1,
        a_endo_con, c_endo_con, a_con, Ψ1grid, achange, corefactor,
        Ψ, Ψ1, Ψ2, c, uc, uce, Va, Vb, EVa, EVb, Wb, Wratio, D, Dendo, Dlast)
end

endostates(::TwoAssetHousehold) = (:aproc, :bproc)
endopolicies(::TwoAssetHousehold) = (aproc=:a, bproc=:b)
exogstates(::TwoAssetHousehold) = (:eproc,)
valuevars(::TwoAssetHousehold) = (:Va, :Vb)
policies(::TwoAssetHousehold) = (:a, :b, :c, :uce, :Ψ)
backwardtargets(::TwoAssetHousehold) = (:a, :b)

function backward_init!(h::TwoAssetHousehold, zeratio, β, eis, rb, ra, χ0, χ1, χ2)
    agrid = grid(h.aproc)
    bgrid = grid(h.bproc)'
    h.Va .= (0.6 .+ agrid .+ 1.1.*bgrid).^(-1/eis)
    h.Vb .= (0.5 .+ 1.2.*agrid .+ bgrid).^(-1/eis)
end

@inline function getΨ(ap::Real, a::Real, ra, χ0, χ1, χ2)
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = abs(a_change)
    sign_change = sign(a_change)

    adj_denominator = a_with_return + χ0
    core_factor = (abs_a_change / adj_denominator)^(χ2-1)

    Ψ = χ1 / χ2 * abs_a_change * core_factor
    Ψ1 = χ1 * sign_change * core_factor
    Ψ2 = -(1 + ra) * (Ψ1 + (χ2 - 1) * Ψ / adj_denominator)
    return Ψ, Ψ1, Ψ2
end

@inline function marginal_cost_grid!(h::TwoAssetHousehold, ra, χ0, χ1, χ2)
    agrid = grid(h.aproc)
    N = length(agrid)
    @inbounds for j in 1:N
        a = agrid[j]
        @simd for i in 1:N
            # Add 1 in advance for later use
            h.Ψ1grid[i,j] = 1 + getΨ(agrid[i], a, ra, χ0, χ1, χ2)[2]
        end
    end
end

@inline function setmin2!(a, a_con, b, bmin)
    @inbounds for i in eachindex(b)
        bi = b[i]
        if bi < bmin
            b[i] = bmin
            a[i] = a_con[i]
        end
    end
end

@inline function updateΨ!(h::TwoAssetHousehold, ra, χ0, χ1, χ2)
    agrid = grid(h.aproc)
    Na, Nb, Ne = size(h.a)
    @inbounds for k in 1:Ne
        for j in 1:Nb
            @simd for i in 1:Na
                Ψ, _, Ψ2 = getΨ(h.a[i,j,k], agrid[i], ra, χ0, χ1, χ2)
                h.Ψ[i,j,k] = Ψ
                h.Ψ2[i,j,k] = Ψ2
            end
        end
    end
end

@inline function setlhs_con!(h::TwoAssetHousehold)
    Na, Nκ, Ne = size(h.lhs_con)
    @inbounds for k in 1:Ne
        for j in 1:Nκ
            κ = h.κgrid[j]
            @simd for i in 1:Na
                h.lhs_con[i,j,k] = h.Wratio[i,1,k] / (1 + κ)
            end
        end
    end
end

function backward_endo!(h::TwoAssetHousehold, EVa, EVb, zeratio, β, eis, rb, ra, χ0, χ1, χ2)
    h.zgrid .= zeratio .* grid(h.eproc)
    Nb = length(grid(h.bproc))
    Ne = length(grid(h.eproc))
    Nκ = length(h.κgrid)
    agrid = grid(h.aproc)
    bgrid = grid(h.bproc)
    bgrid2 = reshape(bgrid, 1, Nb)
    zgrid3 = reshape(h.zgrid, 1, 1, Ne)
    egrid3 = reshape(grid(h.eproc), 1, 1, Ne)

    h.Wb .= β .* EVb
    h.Wratio .= β .* EVa ./ h.Wb

    marginal_cost_grid!(h, ra, χ0, χ1, χ2)
    vli0s23 = splitdimsview(h.li0, (2,3))
    vlp0s23 = splitdimsview(h.lp0, (2,3))
    vWratios = splitdimsview(h.Wratio, (2,3))
    va_endo_uncs23 = splitdimsview(h.a_endo_unc, (2,3))
    vc_endo_uncs = splitdimsview(h.c_endo_unc, (2,3))
    vWbs = splitdimsview(h.Wb, (2,3))
    @inbounds for k in eachindex(vli0s23)
        lhs_equals_rhs_interpolate!(vli0s23[k], vlp0s23[k], vWratios[k], h.Ψ1grid)
        apply_coord!(va_endo_uncs23[k], agrid, vli0s23[k], vlp0s23[k])
        apply_coord!(vc_endo_uncs[k], vWbs[k], vli0s23[k], vlp0s23[k])
    end

    h.c_endo_unc .= h.c_endo_unc.^(-eis)
    h.b_endo .= (h.c_endo_unc .+ h.a_endo_unc .- (1.0.+ra).*agrid .+ bgrid2 .- zgrid3 .+
        getindex.(getΨ.(h.a_endo_unc, agrid, ra, χ0, χ1, χ2), 1)) ./ (1.0.+rb)

    vli0s13 = splitdimsview(h.li0, (1,3))
    vlp0s13 = splitdimsview(h.lp0, (1,3))
    vb_endos = splitdimsview(h.b_endo, (1,3))
    va_uncs = splitdimsview(h.a, (1,3))
    va_endo_uncs13 = splitdimsview(h.a_endo_unc, (1,3))
    vb_uncs = splitdimsview(h.b, (1,3))
    @inbounds for k in eachindex(vli0s13)
        interpolate_coord!(vli0s13[k], vlp0s13[k], bgrid, vb_endos[k])
        apply_coord!(va_uncs[k], va_endo_uncs13[k], vli0s13[k], vlp0s13[k])
        apply_coord!(vb_uncs[k], bgrid, vli0s13[k], vlp0s13[k])
    end

    setlhs_con!(h)

    vli1s23 = splitdimsview(h.li1, (2,3))
    vlp1s23 = splitdimsview(h.lp1, (2,3))
    vlhs_cons = splitdimsview(h.lhs_con, (2,3))
    va_endo_cons = splitdimsview(h.a_endo_con, (2,3))
    vc_endo_cons = splitdimsview(h.c_endo_con, (2,3))
    for k in eachindex(vli1s23)
        lhs_equals_rhs_interpolate!(vli1s23[k], vlp1s23[k], vlhs_cons[k], h.Ψ1grid)
        apply_coord!(va_endo_cons[k], agrid, vli1s23[k], vlp1s23[k])
    end
    for k in 1:Ne
        vWb = view(h.Wb,:,1,k)
        for j in 1:Nκ
            apply_coord!(vc_endo_cons[j,k], vWb, vli1s23[j,k], vlp1s23[j,k])
        end
    end
    h.c_endo_con .= ((1.0.+h.κgrid').*h.c_endo_con).^(-eis)  

    h.b_endo .= (h.c_endo_con .+ h.a_endo_con .- (1.0.+ra).*agrid .+ bgrid[1] .-
        zgrid3 .+ getindex.(getΨ.(h.a_endo_con, agrid, ra, χ0, χ1, χ2), 1)) ./ (1.0.+rb)

    va_cons = splitdimsview(h.a_con, (1,3))
    va_endo_con13s = splitdimsview(h.a_endo_con, (1,3))
    vb_endos = splitdimsview(h.b_endo, (1,3))
    @inbounds for k in eachindex(va_cons)
        interpolate_y!(va_cons[k], bgrid, va_endo_con13s[k], vb_endos[k])
    end

    setmin2!(h.a, h.a_con, h.b, bgrid[1])
    updateΨ!(h, ra, χ0, χ1, χ2)
    h.c .= (1.0.+ra).*agrid .+ (1.0.+rb).*bgrid2 .+ zgrid3 .- h.Ψ .- h.a .- h.b
    h.uc .= h.c.^(-1/eis)
    h.uce .= egrid3 .* h.uc

    h.Va .= (1.0 .+ ra .- h.Ψ2) .* h.uc
    h.Vb .= (1.0 .+ rb) .* h.uc
end

function twoassethhblock(amax, bmax, κmax, Na, Nb, Ne, Nκ, ρe, σe; kwargs...)
    tahh = TwoAssetHousehold(amax, bmax, κmax, Na, Nb, Ne, Nκ, ρe, σe)
    ins = (:zeratio, :β, :eis, :rb, :ra, :χ0, :χ1, :χ2)
    return block(tahh, ins, (:A, :B, :C, :UCE, :CHI); kwargs...)
end

# Additional helper block that are not present in the original Python package
# Allows mapping egrid to zgrid
@simple function income(tax, w, N)
    zeratio = (1 - tax) * w * N
    return zeratio
end

@implicit function pricing(pip=0.1, mc=0.985, r=0.0125, Y=1, κp=0.1, mup=1.015228426395939)
    nkpc = κp*(mc-1/mup) + lead(Y)/Y*log(1+lead(pip))/(1+lead(r)) - log(1+pip)
    return pip, nkpc, Roots_Default
end

@implicit ssargs=(:x0=>(5,15),) function arbitrage(p=10, div=0.14, r=0.0125)
    equity = lead(div) + lead(p) - p * (1 + lead(r))
    return p, equity, Main.Brent
end

@simple function labor(Y, w, K, Z, α)
    N = (Y / Z / lag(K)^α)^(1/(1-α))
    mc = w * N / (1-α) / Y
    return N, mc
end

@simple function investment(Q, K, r, N, mc, Z, δ, εI, α)
    inv = (K / lag(K)-1) / (δ*εI) + 1 - Q
    val = α * lead(Z) * (lead(N)/K)^(1-α) * lead(mc) -
        (lead(K)/K - (1-δ) + (lead(K)/K-1)^2 / (2*δ*εI)) +
        lead(K)/K*lead(Q) - (1+lead(r))*Q
    return inv, val
end

function production_block()
    calis = [:Y=>1, :w=>0.6, :Z=>0.4677898145312322, :α=>0.3299492385786802,
        :r=>0.0125, :δ=>0.02, :εI=>4]
    return block([labor_block(), investment_block()],
        [:Y, :w, :Z, :r], [:Q, :K, :N, :mc],
        calis, [:Q=>2, :K=>11], [:inv, :val].=>0.0, solver=GSL_Hybrids)
end

@simple function dividend(Y, w, N, K, pip, mup, κp, δ, εI)
    ψp = mup / (mup - 1) / 2 / κp * log(1 + pip)^2 * Y
    k_adjust = lag(K) * (K / lag(K) - 1)^2 / (2 * δ * εI)
    I = K - (1 - δ) * lag(K) + k_adjust
    div = Y - w * N - I - ψp
    return ψp, I, div
end

@simple function taylor(rstar, pip, φ)
    i = rstar + φ * pip
    return i
end

@simple function fiscal(r, w, N, G, Bg)
    tax = (r * Bg + G) / w / N
    return tax
end

@simple function finance(i, p, pip, r, div, ω, pshare)
    rb = r - ω
    ra = lag(pshare) * (div + p) / lag(p) + (1 - lag(pshare)) * (1 + r) - 1
    fisher = 1 + lag(i) - (1 + r) * (1 + pip)
    return rb, ra, fisher
end

@simple function wage(pip, w)
    piw = (1 + pip) * w / lag(w) - 1
    return piw
end

@simple function union(piw, N, tax, w, UCE, κw, muw, vφ, frisch, β)
    wnkpc = κw * (vφ * N^(1 + 1/frisch) - (1-tax)*w*N*UCE/muw) + β*log(1+lead(piw)) - log(1+piw)
    return wnkpc
end

@simple function mkt_clearing(p, A, B, Bg, C, I, G, CHI, ψp, ω, Y)
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + ψp + ω * B - Y
    return asset_mkt, wealth, goods_mkt
end

@simple function share_value(p, tot_wealth, Bh)
    pshare = p / (tot_wealth - Bh)
    return pshare
end

@simple function partial_ss(Y, N, K, r, tot_wealth, Bg, δ)
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc
    α = (r + δ) * K / Y / mc
    Z = Y * K^(-α) * N^(α - 1)
    w = mc * (1 - α) * Y / N
    return p, mc, mup, α, Z, w
end

@simple function union_ss(tax, w, UCE, N, muw, frisch)
    vφ = (1 - tax) * w * UCE / muw / N^(1 + 1/frisch)
    wnkpc = vφ * N^(1 + 1/frisch) - (1 - tax) * w * UCE / muw
    return vφ, wnkpc
end

function twoassetmodelss()
    bhh = twoassethhblock(4000, 50, 1, 70, 50, 3, 50, 0.966, 0.92)
    m = model([bhh, income_block(), partial_ss_block(), union_ss_block(), dividend_block(),
        taylor_block(), fiscal_block(), share_value_block(), finance_block(),
        mkt_clearing_block()])
    return m
end

function twoassetmodel()
    bhh = twoassethhblock(4000, 50, 1, 70, 50, 3, 50, 0.966, 0.92)
    m = model([bhh, income_block(), production_block(), pricing_block(), arbitrage_block(),
        dividend_block(), taylor_block(), fiscal_block(), share_value_block(), finance_block(),
        wage_block(), union_block(), mkt_clearing_block()])
    return m
end

end
