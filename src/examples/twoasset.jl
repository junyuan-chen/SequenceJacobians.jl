module TwoAsset

using ..SequenceJacobians

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
        [:Y, :w, :Z, :r], [:Q, :K], [:inv, :val],
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
    fisher = 1 + lead(i) - (1 + r) * (1 + pip)
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


end
