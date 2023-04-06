module SmetsWouters

using ..SequenceJacobians
using Distributions

export swparams, swmodelss, swshocks, swpriors

# Initial values for parameters will be replaced in the full model
@implicit function household(c=0, n=0, r=0, εb=0, c1=0.5, c2=0.5, c3=0.5)
    euler_r = c1 * lag(c) + (1 - c1) * lead(c) + c2 * (n - lead(n)) - c3 * r + εb - c
    return c, euler_r, Roots_Default
end

@implicit function household_f(cf=0, nf=0, rf=0, εb=0, c1=0.5, c2=0.5, c3=0.5)
    euler_r_f = c1 * lag(cf) + (1 - c1) * lead(cf) + c2 * (nf - lead(nf)) - c3 * rf + εb - cf
    return cf, euler_r_f, Roots_Default
end

@simple function wage_markup(c, n, w, λc, γ, σl)
    μw = w - (σl * n + (c - lag(c) * λc / γ) / (1 - λc / γ))
    return μw
end

@simple function wage_markup_f(cf, nf, wf, λc, γ, σl)
    μw_f = wf - (σl * nf + (cf - lag(cf) * λc / γ) / (1 - λc / γ))
    return μw_f
end

@simple function investment(I, q, εI, I1, I2)
    I_r = I1 * lag(I) + (1 - I1) * lead(I) + I2 * q + εI - I
    return I_r
end

@simple function investment_f(If, qf, εI, I1, I2)
    I_r_f = I1 * lag(If) + (1 - I1) * lead(If) + I2 * qf + εI - If
    return I_r_f
end

@simple function tobinq(q, rk, r, εb, q1, c3)
    q_r = q1 * lead(q) + (1 - q1) * lead(rk) - r + εb / c3 - q
    return q_r
end

@simple function tobinq_f(qf, rkf, rf, εb, q1, c3)
    q_r_f = q1 * lead(qf) + (1 - q1) * lead(rkf) - rf + εb / c3 - qf
    return q_r_f
end

@simple function kutilization(rk, z1)
    z = z1 * rk
    return z
end

@simple function kutilization_f(rkf, z1)
    zf = z1 * rkf
    return zf
end

@simple function keffective(z, k)
    ks = z + lag(k)
    return ks
end

@simple function keffective_f(zf, kf)
    ksf = zf + lag(kf)
    return ksf
end

@simple function production(ks, n, w, εa, φp, α)
    y = (1 + φp) * (α * ks + (1 - α) * n + εa)
    μp = α * (ks - n) + εa - w
    return y, μp
end

@simple function production_f(ksf, nf, wf, εa, φp, α)
    yf = (1 + φp) * (α * ksf + (1 - α) * nf + εa)
    μp_f = α * (ksf - nf) + εa - wf
    return yf, μp_f
end

@simple function rentalrate(rk, ks, n, w)
    rk_r = n - ks + w - rk
    return rk_r
end

@simple function rentalrate_f(rkf, ksf, nf, wf)
    rk_r_f = nf - ksf + wf - rkf
    return rk_r_f
end

@simple function capital(k, I, εI, k1, k2)
    k_r = k1 * lag(k) + (1 - k1) * I + k2 * εI - k
    return k_r
end

@simple function capital_f(kf, If, εI, k1, k2)
    k_r_f = k1 * lag(kf) + (1 - k1) * If + k2 * εI - kf
    return k_r_f
end

function firm_blk(solver)
    inits = [:I, :q, :rk, :k]
    tars = [:I_r, :q_r, :rk_r, :k_r]
    blks = [investment_blk(), tobinq_blk(), kutilization_blk(), keffective_blk(),
        production_blk(), rentalrate_blk(), capital_blk()]
    outs = vcat(inits, [:ks, :z, :y, :μp])
    # Expose all inputs to the full model
    ins = setdiff!(union(Iterators.flatten((inputs(x) for x in blks))), outs)
    return block(blks, ins, outs, ins.=>0.5, inits.=>0, tars.=>0, solver=solver)
end

function firm_f_blk(solver)
    inits = [:If, :qf, :rkf, :kf]
    tars = [:I_r_f, :q_r_f, :rk_r_f, :k_r_f]
    blks = [investment_f_blk(), tobinq_f_blk(), kutilization_f_blk(), keffective_f_blk(),
        production_f_blk(), rentalrate_f_blk(), capital_f_blk()]
    outs = vcat(inits, [:ksf, :zf, :yf, :μp_f])
    ins = setdiff!(union(Iterators.flatten((inputs(x) for x in blks))), outs)
    return block(blks, ins, outs, ins.=>0.5, inits.=>0, tars.=>0, solver=solver)
end

@implicit function nkpc_p(πp=0, μp=0, εp=0, π1=0.5, π2=0.5, π3=0.5)
    π_r = π1 * lag(πp) + π2 * lead(πp) - π3 * μp + εp - πp
    return πp, π_r, Roots_Default
end

@implicit function nkpc_w(wout=0, πp=0, μw=0, εw=0, w1=0.5, w2=0.5, w3=0.5, w4=0.5)
    wout_r = w1 * lag(wout) + (1 - w1) * (lead(wout) + lead(πp)) - w2 * πp + w3 * lag(πp) -
        w4 * μw + εw - wout
    return wout, wout_r, Roots_Default
end

@implicit function monetary(i=0, πp=0, y=0, yf=0, εi=0, ρ=0.5, ψ1=0.5, ψ2=0.5, ψ3=0.5)
    taylor_r = ρ * lag(i) + (1 - ρ) * (ψ1 * πp + ψ2 * (y - yf)) + ψ3 * (y - lag(y) -
        yf + lag(yf)) + εi - i
    return i, taylor_r, Roots_Default
end

@simple function fisher(i, r, πp)
    fisher_r = i - (r + lead(πp))
    return fisher_r
end

@simple function labor_mkt(w, wout)
    w_r = w - wout
    return w_r
end

@simple function goods_mkt(y, c, I, z, εg, cy, Iy, zy)
    goods_mkt_r = cy * c + Iy * I + zy * z + εg - y
    return goods_mkt_r
end

@simple function goods_mkt_f(yf, cf, If, zf, εg, cy, Iy, zy)
    goods_mkt_r_f = cy * cf + Iy * If + zy * zf + εg - yf
    return goods_mkt_r_f
end

@simple function ygrowth(y)
    dy = y - lag(y)
    return dy
end

@simple function cgrowth(c)
    dc = c - lag(c)
    return dc
end

@simple function Igrowth(I)
    dI = I - lag(I)
    return dI
end

@simple function wgrowth(w)
    dw = w - lag(w)
    return dw
end

const default_params = (
    σc=1.321,   # elasticity of intertemporal substitution
    σl=2.45,    # elasticity of labor supply with respect to real wage
    cβ=0.114,   # 100(1/β-1)
    λc=0.804,   # external habit for consumption
    α=0.197,    # capital share in production
    φp=0.77,    # fixed cost of production
    ψ=0.401,    # normalized capital utilization adjustment costs elasticity
    φ=6.556,    # steady-state elasticity of capital adjustment cost
    δ=0.025,    # depreciation rate
    ιp=0.174,   # degree of price indexation
    ζp=0.518,   # price rigidity
    ιw=0.536,   # degree of wage indexation
    ζw=0.765,   # wage rigidity
    μw_ss=1.5,  # steady-state wage markup
    curvp=10,   # curvature of Kimball goods market aggregator
    curvw=10,   # curvature of Kimball labor market aggregator
    ρ=0.875,    # Taylor rule inertia
    ψ1=1.866,   # Taylor rule coefficient on inflation
    ψ2=0.12,    # Taylor rule coefficient on output gap
    ψ3=0.125,   # Taylor rule coefficient on growth of output gap
    g=0.18,     # government expenditure share in GDP
    γbar=0.509, # quarterly trend growth rate
    πbar=0.635  # quarterly steady-state inflation rate
)

function swparams(calis)
    v = Dict{Symbol,Float64}(pairs(calis))
    v[:β] = 1 / (v[:cβ] / 100 + 1)
    v[:γ] = 1 + v[:γbar] / 100
    v[:βbar] = v[:β] * v[:γ]^(1 - v[:σc])
    v[:Rkss] = v[:γ]^v[:σc] / v[:β] - (1 - v[:δ])
    v[:Ik] = (1 - (1 - v[:δ]) / v[:γ]) * v[:γ]
    v[:wss] = (v[:α]^v[:α] * (1-v[:α])^(1-v[:α]) /
        ((1+v[:φp]) * v[:Rkss]^v[:α]))^(1/(1-v[:α]))
    v[:nk] = ((1 - v[:α]) / v[:α]) * (v[:Rkss] / v[:wss])
    v[:ky] = (1 + v[:φp]) * v[:nk]^(v[:α] - 1)
    v[:Iy] = v[:Ik] * v[:ky]
    v[:cy] = 1 - v[:g] - v[:Iy]
    v[:zy] = v[:Rkss] * v[:ky]
    v[:whlc] = (1/v[:μw_ss]) * (1-v[:α]) / v[:α] * v[:Rkss] * v[:ky] / v[:cy]
    v[:c1] = (v[:λc] / v[:γ]) / (1 + v[:λc] / v[:γ])
    v[:c2] = ((v[:σc] - 1) * v[:whlc]) / (v[:σc] * (1 + v[:λc] / v[:γ]))
    v[:c3] = (1 - v[:λc] / v[:γ]) / ((1 + v[:λc] / v[:γ]) * v[:σc])
    v[:I1] = 1 / (1 + v[:βbar])
    v[:I2] = v[:I1] / (v[:γ]^2 * v[:φ])
    v[:q1] = v[:βbar] * (1 - v[:δ]) / v[:γ]
    v[:z1] = (1 - v[:ψ]) / v[:ψ]
    v[:k1] = (1 - v[:δ]) / v[:γ]
    v[:k2] = (1 - (1 - v[:δ]) / v[:γ]) * v[:γ]^2 * v[:φ]
    v[:π1] = v[:ιp] / (1 + v[:βbar] * v[:ιp])
    v[:π2] = v[:βbar] / (1 + v[:βbar] * v[:ιp])
    v[:π3] = (1 - v[:βbar] * v[:ζp]) * (1 - v[:ζp]) / (
        v[:ζp] * (v[:φp] * v[:curvp] + 1) * (1 + v[:βbar] * v[:ιp]))
    v[:w1] = 1 / (1 + v[:βbar])
    v[:w2] = (1 + v[:βbar] * v[:ιw]) * v[:w1]
    v[:w3] = v[:ιw] * v[:w1]
    v[:w4] = (1 - v[:βbar] * v[:ζw]) * (1 - v[:ζw]) / (
        (1 + v[:βbar]) * v[:ζw] * ((v[:μw_ss] - 1) * v[:curvw] + 1))
    v[:ibar] = 100 * ((1 + v[:πbar] / 100) / (v[:β] * v[:γ]^(-v[:σc])) - 1)
    v[:i] = v[:ibar]

    # Fill in zero deviations in steady state
    ks = [:I, :If, :q, :qf, :k, :kf, :rk, :rkf, :r, :rf, :n, :nf, :w, :wf, :c, :cf,
        :y, :yf, :πp, :μp, :μp_f, :μw, :μw_f, :z, :zf, :εa, :εb, :εg, :εI, :εi, :εp, :εw]
    for k in ks
        v[k] = 0
    end
    return v
end

function swmodelss(calis, solver)
    m = model([household_blk(), household_f_blk(), wage_markup_blk(), wage_markup_f_blk(),
        firm_blk(solver), firm_f_blk(solver), nkpc_p_blk(), nkpc_w_blk(), monetary_blk(),
        fisher_blk(), labor_mkt_blk(), goods_mkt_blk(), goods_mkt_f_blk(),
        ygrowth_blk(), cgrowth_blk(), Igrowth_blk(), wgrowth_blk()])
    ss = SteadyState(m, calis)
    return m, ss
end

function swshocks()
    ar1s = (:a, :b, :g, :I, :i)
    shs = ShockProcess[ar1shock(Symbol(:σ,n), Symbol(:ar,n), Symbol(:ε,n)) for n in ar1s]
    armas = (arma11shock(Symbol(:σ,n), Symbol(:ar,n), Symbol(:ma,n), Symbol(:ε,n))
        for n in (:p, :w))
    append!(shs, armas)
    return shs
end

function swpriors()
    iΓ = InverseGamma(2.0025, 0.10025)
    B = Beta(2.625, 2.625)
    ns = (:a, :b, :g, :I, :i, :p, :w)
    priors1 = (Symbol(:σ,n)=>iΓ for n in ns)
    priors2 = (Symbol(:ar,n)=>B for n in ns)
    priors3 = (Symbol(:ma,n)=>B for n in (:p, :w))
    priors = [priors1..., priors2..., priors3...]
    return priors
end

end
