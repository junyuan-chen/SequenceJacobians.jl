# Steady state values from the original Python package
# Residuals are rounded to zero
const tassvals = (β = 0.9762739008880043, eis = 0.5, χ0 = 0.25, χ1 = 6.416419594214107,
    χ2 = 2, w = 0.66, ρ_z = 0.966, σ_z = 0.92, Y = 1.0, Z = 0.4677898145312322,
    α = 0.3299492385786802, r = 0.0125, δ = 0.02, εI = 4, κp = 0.1,
    mup = 1.015228426395939, rstar = 0.0125, φ = 1.5, G = 0.2, Bg = 2.8, tot_wealth = 14,
    Bh = 1.04, ω = 0.005, κw = 0.1, muw = 1.1, vφ = 1.713475944050497,
    frisch = 1.0, Q = 1.0, K = 10.0, N = 1.0, mc = 0.985, inv = 0.0, val = 0.0,
    pip = 0.0, nkpc = 0.0, piw = 0.0, i = 0.012499999999379603, ψp = 0.0,
    I = 0.2, div = 0.14, p = 11.2, equity = 0.0, pshare = 0.864197530864201,
    rb = 0.0075, ra = 0.0125, fisher = 0.0, tax = 0.3560606060606061,
    A = 12.96, B = 1.04, C = 0.5820937276337765, UCE = 4.434878914013179,
    CHI = 0.012706305302404835, wnkpc = 0.0, asset_mkt = 0.0, wealth = 14.0, goods_mkt = 0.0,
    zeratio=0.425)

compare(a::NT, b::NT, tol::Real) where NT<:NamedTuple =
    all(k->isapprox(a[k], b[k], atol=tol), keys(a))

@testset "TwoAsset" begin
    using SequenceJacobians: TwoAsset as ta
    @testset "SimpleBlock" begin
        bdividend = ta.dividend_blk()
        @test compare(steadystate!(bdividend, tassvals), tassvals, 1e-8)
        btaylor = ta.taylor_blk()
        @test compare(steadystate!(btaylor, tassvals), tassvals, 1e-8)
        bfiscal = ta.fiscal_blk()
        @test compare(steadystate!(bfiscal, tassvals), tassvals, 1e-8)
        bfinance = ta.finance_blk()
        @test compare(steadystate!(bfinance, tassvals), tassvals, 1e-8)
        bwage = ta.wage_blk()
        @test compare(steadystate!(bwage, tassvals), tassvals, 1e-8)
        bunion = ta.union_blk()
        @test compare(steadystate!(bunion, tassvals), tassvals, 1e-8)
        bmkt_clearing = ta.mkt_clearing_blk()
        @test compare(steadystate!(bmkt_clearing, tassvals), tassvals, 1e-7)
        bshare_value = ta.share_value_blk()
        @test compare(steadystate!(bshare_value, tassvals), tassvals, 1e-8)
        bpartial_ss = ta.partial_ss_blk()
        @test compare(steadystate!(bpartial_ss, tassvals), tassvals, 1e-7)
        bunion_ss = ta.union_ss_blk()
        @test compare(steadystate!(bunion_ss, tassvals), tassvals, 1e-8)
    end

    @testset "CombinedBlock" begin
        bpricing = ta.pricing_blk()
        @test compare(steadystate!(bpricing, tassvals), tassvals, 1e-8)
        barbitrage = ta.arbitrage_blk()
        @test compare(steadystate!(barbitrage, tassvals), tassvals, 1e-8)
        bproduction = ta.production_blk()
        @test compare(steadystate!(bproduction, tassvals), tassvals, 1e-8)
    end

    @testset "HetBlock" begin
        # Compare results for a single backward iteration
        h = ta.TwoAssetHousehold(4000, 50, 1, 70, 50, 3, 50, 0.966, 0.92)
        invals = map(k->tassvals[k], (:zeratio, :β, :eis, :rb, :ra, :χ0, :χ1, :χ2))
        backward_init!(h, invals...)
        backward!(h, invals...)
        @test h.zgrid ≈ [0.07784149, 0.28593115, 1.05029622] atol=1e-8
        @test h.Va[1:3] ≈ [167.098583, -2.95289932, -3.74178335] atol=1e-6
        @test h.Va[end-2:end] ≈ [-1.14524645e-6, -9.31948487e-7, -7.51479291e-7] atol=1e-14
        @test h.Vb[1:3] ≈ [166.273405, 100.903633, 94.6001939] atol=1e-6
        @test h.Vb[end-2:end] ≈ [1.28353202e-5, 1.01973016e-5, 8.05678409e-6] atol=1e-13

        b = ta.twoassethhblock(4000, 50, 1, 70, 50, 3, 50, 0.966, 0.92)
        varvals = steadystate!(b, tassvals)
        @test compare(varvals, tassvals, 1e-6)
        # Compare results with block-level steady state from original Python package
        @test varvals[:A] ≈ 12.959999999999448 atol=1e-6
        @test varvals[:B] ≈ 1.0400000000001128 atol=1e-7
        @test varvals[:C] ≈ 0.5820937276337765 atol=1e-8
        @test varvals[:UCE] ≈ 4.434878914013179 atol=1e-6
        @test varvals[:CHI] ≈ 0.012706305302404835 atol=1e-9

        # Compare results with original Python package
        # Some small discrepancies for the anticipation effects are expected
        # due to the different epsilons for finite differencing
        j = jacobian(b, Val(4), 5, tassvals)
        # dA/drb
        @test j.Js[4][1][1,:] ≈ [0.01910664, -0.75874561, -0.68216018, -0.61887753, -0.56469997] atol=1e-2
        @test j.Js[4][1][5,:] ≈ [0.09487919, -0.47825256, -1.05770784, -1.64550569, -2.24358476] atol=1e-2
        # dB/drb
        @test j.Js[4][2][1,:] ≈ [0.99425083, 0.8778591 , 0.79129299, 0.71896877, 0.65654112] atol=1e-2
        @test j.Js[4][2][5,:] ≈ [0.84871964, 1.51917745, 2.19827569, 2.88869726, 3.59303761] atol=1e-2
        # dC/drb
        @test j.Js[4][3][1,:] ≈ [0.02611385, -0.12006327, -0.10602823, -0.09419434, -0.08394261] atol=1e-3
        @test j.Js[4][3][5,:] ≈ [0.02397805,  0.03287429,  0.04180844,  0.05085865,  0.06011889] atol=1e-3
        # dUCE/drb
        @test j.Js[4][4][1,:] ≈ [-0.37965504, 1.72651783, 1.53462033, 1.37684598, 1.2425804] atol=1e-1
        @test j.Js[4][4][5,:] ≈ [-0.32978632, -0.45494228, -0.57883471, -0.70318167, -0.8296967] atol=1e-2
        # dCHI/drb
        @test j.Js[4][5][1,:] ≈ [0.00052872, 0.00098339, -0.00307608, -0.00587232, -0.00787712] atol=1e-3
        @test j.Js[4][5][5,:] ≈ [0.00021719, -0.00206191, -0.00427481, -0.00644023, -0.00859005] atol=1e-4

        j = jacobian(b, Val(5), 5, tassvals)
        # dA/dra
        @test j.Js[5][1][5,:] ≈ [12.55876212, 13.12224921, 13.69123629, 14.26733031, 14.8520677] atol=1e-2
        # dB/dra
        @test j.Js[5][2][5,:] ≈ [-0.24164763, -0.79501758, -1.35692425, -1.92922367, -2.51371582] atol=1e-2
        # dC/dra
        @test j.Js[5][3][5,:] ≈ [0.2380636 , 0.23233434, 0.22654506, 0.22063715, 0.220252] atol=1e-3
        # dUCE/dra
        @test j.Js[5][4][5,:] ≈ [-2.05280539, -2.00079259, -1.94985638, -1.89899934, -1.8285617] atol=1e-2
        # dCHI/dra
        @test j.Js[5][5][5,:] ≈ [0.01076234, 0.01300736, 0.015212, 0.01739816, 0.02134768] atol=1e-4
    end

    @testset "SteadyState Jacobian" begin
        Main.backwardsolver(::ta.TwoAssetHousehold) = NLsolve_anderson
        mss = ta.twoassetmodelss()
        mss.pool[1].ssargs[:backaastart] = 100
        calis = Dict{Symbol,Float64}(:Y=>1, :N=>1, :K=>10, :r=>0.0125, :rstar=>0.0125,
        :tot_wealth=>14, :δ=>0.02, :pip=>0, :κp=>0.1, :muw=>1.1, :Bh=>1.04, :Bg=>2.8, :G=>0.2,
        :eis=>0.5, :frisch=>1, :χ0=>0.25, :χ2=>2, :εI=>4, :ω=>0.005, :κw=>0.1, :φ=>1.5)
        inits = [:β=>0.976, :χ1=>6.5]
        tars = [:asset_mkt=>0, :B=>1.04]
        ss = SteadyState(mss, calis, inits, tars)
        @time solve!(GSL_Hybrids, ss, xtol=1e-10)
        # Compare results with original Python package
        @test getval(ss, :A) ≈ 12.96 atol=1e-5
        @test getval(ss, :B) ≈ 1.04 atol=1e-6
        @test getval(ss, :C) ≈ 0.5820937276337765 atol=1e-7
        @test getval(ss, :UCE) ≈ 4.434878914013179 atol=1e-6
        @test getval(ss, :CHI) ≈ 0.012706305302404835 atol=1e-8

        m = ta.twoassetmodel()
        # Directly move the household block
        m.pool[1] = mss.pool[1]
        foreach(i->steadystate!(m.pool[i], tassvals), 3:5)

        # Compute Jacobians with default epsilon from FiniteDiff.jl
        # The default epsilon is much smaller than the one set by the Python package
        @time J = TotalJacobian(m, [:rstar,:Z,:G,:r,:w,:Y], [:asset_mkt,:fisher,:wnkpc],
            tassvals, 300)
        gj = GEJacobian(J, (:rstar,:Z,:G), keepH_U=true)
        # dasset_mkt/dw
        @test gj.H_U[1,1:3] ≈ [0.01679725, 0.04170578, 0.03739954] atol=1e-3
        @test gj.H_U[300,298:300] ≈ [-0.34252791, -0.35230779, -0.36274057] atol=1e-3
        # dasset_mkt/dY
        @test gj.H_U[1,301:303] ≈ [-9.62962356e-1, 6.12458973e-2, 5.67827128e-2] atol=1e-3
        @test gj.H_U[300,598:600] ≈ [-3.40122735e-1, -3.49718623e-1, -3.59957420e-1] atol=1e-3
        # dasset_mkt/dr
        @test gj.H_U[1,601:603] ≈ [-4.16475490e-2, -1.03942468, -9.72823634e-1] atol=1e-2
        @test gj.H_U[300,898:900] ≈ [-1.32624290e1, -1.33262797e1, -1.33873877e1] atol=1e-2
        # dfisher/dw
        @test gj.H_U[301,1:3] ≈ [-1.51107955e-1, -1.46551915e-1, -1.42230528e-1] atol=1e-8
        @test gj.H_U[600,298:300] ≈ [-4.51897529e-3, 2.19083483e-1, 6.52185813e-2] atol=1e-9
        # dfisher/dY
        @test gj.H_U[301,301:303] ≈ [-4.91100852e-2, -4.58536360e-2, -4.28131192e-2] atol=1e-9
        @test gj.H_U[600,598:600] ≈ [-4.45119066e-3,  6.80472306e-2,  1.80456267e-2] atol=1e-9
        # dfisher/dr
        @test gj.H_U[301,601:603] ≈ [-1, -5.46380397e-2, -1.04978532e-1] atol=1e-8
        @test gj.H_U[600,898:900] ≈ [9.17699596e-2, 9.70738865e-2, -9.03064239e-1] atol=1e-8
        # dwnkpc/dw
        @test gj.H_U[601,1:3] ≈ [-3.38779477, 1.49386238, 1.21228316e-2] atol=1e-3
        @test gj.H_U[900,298:300] ≈ [4.40888876e-03,  1.52001088e+00, -1.92003948] atol=1e-3
        # dwnkpc/dY
        @test gj.H_U[601,301:303] ≈ [2.28710864e-1,  2.16437418e-2,  1.91089705e-2] atol=1e-3
        @test gj.H_U[900,598:600] ≈ [4.36845379e-3,  4.81102885e-3,  2.11259782e-1] atol=1e-3
        # dwnkpc/dr
        @test gj.H_U[601,601:603] ≈ [6.91972618e-1, -1.72516350e-1, -1.57461667e-1] atol=1e-2
        @test gj.H_U[900,898:900] ≈ [1.56318555e-1,  1.58314638e-1,  8.90424063e-1] atol=1e-2

        Gyr = getG!(gj, :rstar, :Y)
        @test Gyr[1,1:3] ≈ [-1.12544615, -9.42169946e-1, -6.81522845e-1] atol=1e-3
        @test Gyr[300,298:300] ≈ [-3.17242939e-2, -4.88707368e-2, 0] atol=1e-2
        Gwg = getG!(gj, :G, :w)
        @test Gwg[1,1:3] ≈ [1.42252020e-1, 3.48736221e-2, -3.10755825e-2] atol=1e-3
        @test Gwg[300,298:300] ≈ [1.87887110e-1, 3.06551227e-1, 4.89042824e-1] atol=1e-3
        Grz = getG!(gj, :Z, :r)
        @test Grz[1,1:3] ≈ [3.28290545e-1, 2.89395288e-1, 2.21653282e-1] atol=1e-3
        @test Grz[300,298:300] ≈ [-5.79345415e-2, -4.79892126e-1, -2.01655494e-1] atol=1e-3

        # Recompute the Jacobians with the same epsilon used by the Python package
        m.pool[1].diffargs[] = (twosided=true, epsilon=1e-4)
        J = TotalJacobian(m, [:rstar,:Z,:G,:r,:w,:Y], [:asset_mkt,:fisher,:wnkpc],
            tassvals, 300)
        @test m.pool[1].jacargs[:jacca].epsilon == 1e-4
        gj = GEJacobian(J, (:rstar,:Z,:G), keepH_U=true)

        Gyr = getG!(gj, :rstar, :Y)
        @test Gyr[1,1:3] ≈ [-1.12544615, -9.42169946e-1, -6.81522845e-1] atol=1e-3
        @test Gyr[300,298:300] ≈ [-3.17242939e-2, -4.88707368e-2, 0] atol=1e-3
        Gwg = getG!(gj, :G, :w)
        @test Gwg[1,1:3] ≈ [1.42252020e-1, 3.48736221e-2, -3.10755825e-2] atol=1e-3
        @test Gwg[300,298:300] ≈ [1.87887110e-1, 3.06551227e-1, 4.89042824e-1] atol=1e-3
        Grz = getG!(gj, :Z, :r)
        @test Grz[1,1:3] ≈ [3.28290545e-1, 2.89395288e-1, 2.21653282e-1] atol=1e-3
        @test Grz[300,298:300] ≈ [-5.79345415e-2, -4.79892126e-1, -2.01655494e-1] atol=1e-3
    end
end
