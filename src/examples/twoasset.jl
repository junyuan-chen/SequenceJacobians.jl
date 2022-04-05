module TwoAsset

using ..SequenceJacobians

function pricing(inflat, inflatlead, mc, rlead, Y, Ylead, κp, mup)
    nkpc = κp*(mc-1/mup) + Ylead/Y*log(1+inflatlead)/(1+rlead) - log(1+inflat)
    return nkpc
end

function arbitrage(p, plead, divlead, rlead)
    equity = divlead + plead - p * (1 + rlead)
    return equity
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


end
