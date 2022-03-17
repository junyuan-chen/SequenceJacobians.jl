module RBC

using ..SequenceJacobians

export rbcblocks

function firm(Klag, L, Z, α, δ)
    r = α * Z * (Klag / L)^(α-1.0) - δ
    w = (1.0 - α) * Z * (Klag / L)^α
    Y = Z * Klag^α * L^(1.0 - α)
    return r, w, Y
end

function household(K, Klag, L, w, eis, frisch, φ, δ)
    C = (w / (φ * L^(1.0/frisch)))^eis
    I = K - (1.0 - δ) * Klag
    return C, I
end

function mkt_clearing(r, rlead, C, Clead, Y, I, K, Klag, L, w, eis, β)
    goods_mkt = Y - C - I
    euler = C^(-1.0/eis) - β*(1 + rlead)*Clead^(-1.0/eis)
    walras = C + K - (1 + r)*Klag - w*L
    return goods_mkt, euler, walras
end

rbcsstarget(r, Y) = r, Y

function rbcblocks()
    bfirm = block(firm, [lag(:K), :L, :Z, :α, :δ], [:r, :w, :Y])
    bhh = block(household, [:K, lag(:K), :L, :w, :eis, :frisch, :φ, :δ], [:C, :I])
    bmkt = block(mkt_clearing,
        [:r, lead(:r), :C, lead(:C), :Y, :I, :K, lag(:K), :L, :w, :eis, :β],
        [:goods_mkt, :euler, :walras])
    bss = block(rbcsstarget, [:r, :Y], [:rss, :Yss])
    return bfirm, bhh, bmkt, bss
end

end
