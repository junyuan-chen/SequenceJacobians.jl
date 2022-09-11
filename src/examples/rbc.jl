module RBC

using ..SequenceJacobians

export rbcblocks

@simple function firm(K, L, Z, α, δ)
    r = α * Z * (lag(K) / L)^(α-1) - δ
    w = (1-α) * Z * (lag(K) / L)^α
    Y = Z * lag(K)^α * L^(1-α)
    return r, w, Y
end

@simple function household(K, L, w, eis, frisch, φ, δ)
    C = (w / (φ * L^(1/frisch)))^eis
    I = K - (1 - δ) * lag(K)
    return C, I
end

@simple function mkt_clearing(r, C, Y, I, K, L, w, eis, β)
    goods_mkt = Y - C - I
    euler = C^(-1/eis) - β*(1+lead(r))*lead(C)^(-1/eis)
    walras = C + K - (1+r)*lag(K) - w*L
    return goods_mkt, euler, walras
end

rbcblocks() = (firm_blk(), household_blk(), mkt_clearing_blk())

end
