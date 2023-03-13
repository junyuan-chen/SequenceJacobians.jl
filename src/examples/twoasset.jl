module TwoAsset

using ..SequenceJacobians
using ..SequenceJacobians.ExampleUtils
using LinearAlgebra: mul!
using SplitApplyCombine: splitdimsview

import SequenceJacobians: endoprocs, exogprocs, valuevars, expectedvalues, policies,
    backwardtargets, backward_init!, backward_endo!

export twoassethhblock, twoassetmodelss, twoassetmodel

struct TwoAssetHousehold{TF<:AbstractFloat} <: AbstractHetAgent
    aproc::EndoProc{TF,3}
    bproc::EndoProc{TF,3}
    eproc::ExogProc{TF}
    κgrid::Vector{TF}
    zgrid::Vector{TF}
    li0a::Array{Int,3}
    lp0a::Array{TF,3}
    a_endo_unc::Array{TF,3}
    c_endo_unc::Array{TF,3}
    b_endo::Array{TF,3}
    li0b::Array{Int,3}
    lp0b::Array{TF,3}
    a::Array{TF,3}
    alast::Array{TF,3}
    b::Array{TF,3}
    blast::Array{TF,3}
    icons::Array{Bool,3}
    icons0::Vector{Int}
    icons1::Vector{Int}
    lhs_con::Array{TF,3}
    li1::Array{Int,3}
    lp1::Array{TF,3}
    a_endo_con::Array{TF,3}
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
    li0a = Array{Int,3}(undef, dims0)
    lp0a = Array{Float64,3}(undef, dims0)
    a_endo_unc = similar(lp0a)
    c_endo_unc = similar(lp0a)
    b_endo = similar(lp0a)
    li0b = similar(li0a)
    lp0b = similar(lp0a)
    a = similar(lp0a)
    alast = similar(a)
    b = similar(a)
    blast = similar(a)
    icons = Array{Bool,3}(undef, dims0)
    fill!(icons, false)
    icons0 = ones(Int, Ne)
    icons1 = similar(icons0)
    lhs_con = Array{Float64,3}(undef, dims1)
    li1 = Array{Int,3}(undef, dims1)
    lp1 = Array{Float64,3}(undef, dims1)
    a_endo_con = similar(lp1)
    a_con = similar(lp1)
    Ψ1grid = Matrix{Float64}(undef, Na, Na)
    achange = similar(a)
    corefactor = similar(a)
    Ψ = similar(a)
    Ψ1 = similar(a)
    Ψ2 = similar(a)
    c = similar(a)
    uc = similar(a)
    uce = similar(a)
    Va = similar(a)
    Vb = similar(a)
    EVa = similar(a)
    EVb = similar(a)
    Wb = similar(a)
    Wratio = similar(a)
    D = similar(a)
    Dendo = similar(a)
    Dlast = similar(a)
    return TwoAssetHousehold{eltype(c)}(aproc, bproc, eproc, κgrid, zgrid,
        li0a, lp0a, a_endo_unc, c_endo_unc, b_endo, li0b, lp0b,
        a, alast, b, blast, icons, icons0, icons1, lhs_con, li1, lp1,
        a_endo_con, a_con, Ψ1grid, achange, corefactor,
        Ψ, Ψ1, Ψ2, c, uc, uce, Va, Vb, EVa, EVb, Wb, Wratio, D, Dendo, Dlast)
end

endoprocs(h::TwoAssetHousehold) = (h.aproc, h.bproc)
exogprocs(h::TwoAssetHousehold) = (h.eproc,)
valuevars(h::TwoAssetHousehold) = (h.Va, h.Vb)
expectedvalues(h::TwoAssetHousehold) = (h.EVa, h.EVb)
policies(h::TwoAssetHousehold) = (h.a, h.b, h.c, h.uce, h.Ψ)
backwardtargets(h::TwoAssetHousehold) = (h.a=>h.alast, h.b=>h.blast)

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

@inline function findcons!(icons, icons0, icons1, b, bmin, li0a)
    Na, Nb, Ne = size(b)
    @inbounds for k in 1:Ne
        for j in 1:Nb
            for i in 1:Na
                bi = b[i,j,k]
                if bi < bmin
                    b[i,j,k] = bmin
                    # Indicate whether b is constrained
                    icons[i,j,k] = true
                    # Update the upper bound for indices with constrained b
                    icons0[k] = i
                else
                    icons[i,j,k] = false
                end
            end
        end
        i0 = icons0[k]
        # Obtain the bound for the grid in the next period
        i1 = li0a[i0,1,k]
        i1 < Na && (i1 += 1)
        icons1[k] = i1
    end
end

function setacon!(a, a_con, icons)
    Na, Nb, Ne = size(a)
    @inbounds for k in 1:Ne
        for j in 1:Nb
            for i in 1:Na
                if icons[i,j,k]
                    a[i,j,k] = a_con[i,j,k]
                end
            end
        end
    end
end

# Algorithm based on Auclert, Bardóczy, Rognlie and Straub (2021)
function backward_endo!(h::TwoAssetHousehold, EVa, EVb, zeratio, β, eis, rb, ra, χ0, χ1, χ2)
    h.zgrid .= zeratio .* grid(h.eproc)
    Na = length(grid(h.aproc))
    Nb = length(grid(h.bproc))
    Ne = length(grid(h.eproc))
    Nκ = length(h.κgrid)
    agrid = grid(h.aproc)
    bgrid = grid(h.bproc)
    egrid = grid(h.eproc)
    bmin = bgrid[1]

    h.Wb .= β .* EVb
    h.Wratio .= EVa ./ EVb

    # First find the policy without imposing the constraint on b
    marginal_cost_grid!(h, ra, χ0, χ1, χ2)
    vli0as = splitdimsview(h.li0a, (2,3))
    vlp0as = splitdimsview(h.lp0a, (2,3))
    vWratios = splitdimsview(h.Wratio, (2,3))
    vli0bs = splitdimsview(h.li0b, (1,3))
    vlp0bs = splitdimsview(h.lp0b, (1,3))
    vb_endos = splitdimsview(h.b_endo, (1,3))
    @inbounds Threads.@threads for k in 1:Ne
        zk = h.zgrid[k]
        # Find a', b and c on the (a, b', e) grid
        for j in 1:Nb
            lhs_equals_rhs_interpolate!(vli0as[j,k], vlp0as[j,k], vWratios[j,k], h.Ψ1grid)
            for i in 1:Na
                li0a, lp0a = h.li0a[i,j,k], h.lp0a[i,j,k]
                a_endo_unc = lp0a * agrid[li0a] + (1-lp0a) * agrid[li0a+1]
                h.a_endo_unc[i,j,k] = a_endo_unc
                c_endo_unc = (lp0a * h.Wb[li0a,j,k] + (1-lp0a) * h.Wb[li0a+1,j,k])^(-eis)
                h.c_endo_unc[i,j,k] = c_endo_unc
                h.b_endo[i,j,k] = (c_endo_unc + a_endo_unc - (1+ra)*agrid[i] + bgrid[j] -
                    zk + getΨ(a_endo_unc, agrid[i], ra, χ0, χ1, χ2)[1]) / (1+rb)
            end
        end
        # Find a' and b' on the (a, b, e) grid
        for i in 1:Na
            interpolate_coord!(vli0bs[i,k], vlp0bs[i,k], bgrid, vb_endos[i,k])
        end
        for (i, j) in Base.product(1:Na, 1:Nb)
            li0b, lp0b = h.li0b[i,j,k], h.lp0b[i,j,k]
            h.a[i,j,k] = lp0b * h.a_endo_unc[i,li0b,k] + (1-lp0b) * h.a_endo_unc[i,li0b+1,k]
            h.b[i,j,k] = lp0b * bgrid[li0b] + (1-lp0b) * bgrid[li0b+1]
        end
    end

    # Find the subset of the state space where b is constrained
    findcons!(h.icons, h.icons0, h.icons1, h.b, bmin, h.li0a)

    # Resolve the problems imposing b is binding with different κ
    vli1s = splitdimsview(h.li1, (2,3))
    vlp1s = splitdimsview(h.lp1, (2,3))
    vlhs_cons = splitdimsview(h.lhs_con, (2,3))
    va_cons = splitdimsview(h.a_con, (1,3))
    va_endo_con13s = splitdimsview(h.a_endo_con, (1,3))
    vb_endos = splitdimsview(h.b_endo, (1,3))
    @inbounds Threads.@threads for k in 1:Ne
        iamax0 = h.icons0[k]
        iamax1 = h.icons1[k]
        zk = h.zgrid[k]
        for j in 1:Nκ
            κ = h.κgrid[j]
            for i in 1:iamax1
                h.lhs_con[i,j,k] = h.Wratio[i,1,k] / (1 + κ)
            end
            lhs_equals_rhs_interpolate!(vli1s[j,k], vlp1s[j,k], vlhs_cons[j,k], h.Ψ1grid, iamax1, iamax0)

            # Find a' and c on the subset of (a, κ, e) grid where b is binding
            for i in 1:iamax0
                li1, lp1 = h.li1[i,j,k], h.lp1[i,j,k]
                a_endo_con = lp1*agrid[li1]+(1-lp1)*agrid[li1+1]
                h.a_endo_con[i,j,k] = a_endo_con
                Wb1 = (lp1* h.Wb[li1,1,k] + (1-lp1) * h.Wb[li1+1,1,k])
                c_endo_con = (Wb1*(1+κ))^(-eis)

                # Get b on the (a, κ, e) grid by imposing b'=0
                ap = a_endo_con
                ai = agrid[i]
                h.b_endo[i,j,k] = (c_endo_con + ap - (1+ra)*ai + bmin - zk +
                    getΨ(ap, ai, ra, χ0, χ1, χ2)[1]) / (1+rb)
            end
        end

        # Find a' on the (a, b, e) grid for the constrained cases
        for i in 1:iamax0
            interpolate_y!(va_cons[i,k], bgrid, va_endo_con13s[i,k], vb_endos[i,k])
        end
    end

    # Combine results for the constrained and unconstrained cases
    setacon!(h.a, h.a_con, h.icons)

    @inbounds Threads.@threads for k in 1:Ne
        zk = h.zgrid[k]
        ek = egrid[k]
        for j in 1:Nb
            for i in 1:Na
                Ψ, _, Ψ2 = getΨ(h.a[i,j,k], agrid[i], ra, χ0, χ1, χ2)
                h.Ψ[i,j,k] = Ψ
                h.Ψ2[i,j,k] = Ψ2

                c = (1+ra)*agrid[i] + (1+rb)*bgrid[j] + zk - Ψ - h.a[i,j,k] - h.b[i,j,k]
                h.c[i,j,k] = c
                uc = c^(-1/eis)
                h.uc[i,j,k] = uc
                h.uce[i,j,k] = ek * uc

                h.Va[i,j,k] = (1 + ra - Ψ2) * uc
                h.Vb[i,j,k] = (1 + rb) * uc
            end
        end
    end
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

@implicit function pricing(pip=0.1, mc=0.985, r=0.0125, Y=1, κp=0.1, mup=1.015228426395939, εmup=0.0)
    nkpc = κp*(mc-1/mup) + lead(Y)/Y*log(1+lead(pip))/(1+lead(r)) + εmup - log(1+pip)
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

@simple function investment(Q, K, r, N, mc, Z, δ, εI, α, εr)
    inv = (K / lag(K)-1) / (δ*εI) + 1 - Q
    val = α * lead(Z) * (lead(N)/K)^(1-α) * lead(mc) -
        (lead(K)/K - (1-δ) + (lead(K)/K-1)^2 / (2*δ*εI)) +
        lead(K)/K*lead(Q) - (1+lead(r)+εr)*Q
    return inv, val
end

function production_blk()
    calis = [:Y=>1, :w=>0.6, :Z=>0.4677898145312322, :α=>0.3299492385786802,
        :r=>0.0125, :δ=>0.02, :εI=>4, :εr=>0]
    return block([labor_blk(), investment_blk()],
        [:Y, :w, :Z, :α, :r, :δ, :εI, :εr], [:Q, :K, :N, :mc],
        calis, [:Q=>2, :K=>11], [:inv, :val].=>0.0, solver=GSL_Hybrids)
end

@simple function dividend(Y, w, N, K, pip, mup, κp, δ, εI)
    ψp = mup / (mup - 1) / 2 / κp * log(1 + pip)^2 * Y
    k_adjust = lag(K) * (K / lag(K) - 1)^2 / (2 * δ * εI)
    I = K - (1 - δ) * lag(K) + k_adjust
    div = Y - w * N - I - ψp
    return ψp, I, div
end

@simple function taylor(rstar, pip, Y, φ, φy)
    i = rstar + φ * pip + φy * (Y - 1)
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

@simple function union(piw, N, tax, w, UCE, κw, muw, vφ, frisch, β, εmuw)
    wnkpc = κw * (vφ * N^(1 + 1/frisch) - (1-tax)*w*N*UCE/muw) + β*log(1+lead(piw)) +
        εmuw - log(1+piw)
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
    m = model([bhh, income_blk(), partial_ss_blk(), union_ss_blk(), dividend_blk(),
        taylor_blk(), fiscal_blk(), share_value_blk(), finance_blk(),
        mkt_clearing_blk()])
    return m
end

function twoassetmodel()
    bhh = twoassethhblock(4000, 50, 1, 70, 50, 3, 50, 0.966, 0.92)
    m = model([bhh, income_blk(), production_blk(), pricing_blk(), arbitrage_blk(),
        dividend_blk(), taylor_blk(), fiscal_blk(), share_value_blk(), finance_blk(),
        wage_blk(), union_blk(), mkt_clearing_blk()])
    return m
end

end
