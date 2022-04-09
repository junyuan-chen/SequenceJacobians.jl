module TwoAsset

using ..SequenceJacobians

function pricing(pip, piplead, mc, rlead, Y, Ylead, κp, mup)
    nkpc = κp*(mc-1/mup) + Ylead/Y*log(1+piplead)/(1+rlead) - log(1+pip)
    return nkpc
end

function pricing_block()
    ins = (:mc, lead(:r), :Y, lead(:Y), :κp, :mup)
    outs = :pip
    m = model(block(pricing, (:pip, lead(:pip), ins...), :nkpc))
    ss = SteadyState(m, [:mc=>0.985, :r=>0.0125, :Y=>1, :κp=>0.1, :mup=>1.015228426395939], :pip=>0.1, :nkpc=>0)
    return block(ss, ins, outs, :nkpc, Solver=Roots_Default)
end

function arbitrage(p, plead, divlead, rlead)
    equity = divlead + plead - p * (1 + rlead)
    return equity
end

function arbitrage_block()
    ins = (lead(:div), lead(:r))
    outs = :p
    barbitrage = block(arbitrage, (:p, lead(:p), ins...), :equity)
    return block(barbitrage, ins, outs, :equity, [:div=>0.14, :r=>0.0125], :p=>10, :equity=>0,
        Solver=Brent, ssargs=(:x0=>(5,15),))
end

function labor(Y, w, Klag, Z, α)
    N = (Y / Z / Klag^α)^(1/(1-α))
    mc = w * N / (1-α) / Y
    return N, mc
end

function investment(Q, Qlead, K, Klag, Klead, rlead, Nlead, mclead, Zlead, δ, εI, α)
    inv = (K / Klag-1) / (δ*εI) + 1 - Q
    val = α * Zlead * (Nlead/K)^(1-α) * mclead -
        (Klead/K - (1-δ) + (Klead/K-1)^2 / (2*δ*εI)) +
        Klead/K*Qlead - (1+rlead)*Q
    return inv, val
end

function production_block()
    blabor = block(labor, [:Y, :w, lag(:K), :Z, :α], [:N, :mc])
    binvest = block(investment, [:Q, lead(:Q), :K, lag(:K), lead(:K), lead(:r), lead(:N), lead(:mc), lead(:Z), :δ, :εI, :α], [:inv, :val])
    calis = [:Y, :w, :Z, :α, :r, :δ, :εI]
    return block([blabor, binvest], [:Y, :w, :Z, :r], [:Q, :K], [:inv, :val],
        calis.=>[1.0, 0.66, 0.4677898145312322, 0.3299492385786802, 0.0125, 0.02, 4],
        [:Q=>2, :K=>11], [:inv, :val].=>0.0, Solver=GSL_Hybrids)
end

function dividend(Y, w, N, K, Klag, pip, mup, κp, δ, εI)
    ψp = mup / (mup - 1) / 2 / κp * log(1 + pip)^2 * Y
    k_adjust = Klag * (K / Klag - 1)^2 / (2 * δ * εI)
    I = K - (1 - δ) * Klag + k_adjust
    div = Y - w * N - I - ψp
    return ψp, I, div
end

function taylor(rstar, pip, φ)
    i = rstar + φ * pip
    return i
end

function fiscal(r, w, N, G, Bg)
    tax = (r * Bg + G) / w / N
    return tax
end

function finance(ilag, p, plag, pip, r, div, ω, psharelag)
    rb = r - ω
    ra = psharelag * (div + p) / plag + (1 - psharelag) * (1 + r) - 1
    fisher = 1 + ilag - (1 + r) * (1 + pip)
    return rb, ra, fisher
end

function wage(pip, w, wlag)
    piw = (1 + pip) * w / wlag - 1
    return piw
end

function union(piw, piwlead, N, tax, w, UCE, κw, muw, vφ, frisch, β)
    wnkpc = κw * (vφ * N^(1 + 1/frisch) - (1-tax)*w*N*UCE/muw) + β*log(1+piwlead) - log(1+piw)
    return wnkpc
end

function mkt_clearing(p, A, B, Bg, C, I, G, CHI, ψp, ω, Y)
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + ψp + ω * B - Y
    return asset_mkt, wealth, goods_mkt
end

function share_value(p, tot_wealth, Bh)
    pshare = p / (tot_wealth - Bh)
    return pshare
end



end
