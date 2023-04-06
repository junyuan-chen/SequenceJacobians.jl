# Steady state values from the original Python package
# Residuals are rounded to zero
const tassvals = (β = 0.9762739008880043, eis = 0.5, χ0 = 0.25, χ1 = 6.416419594214107,
    χ2 = 2.0, w = 0.66, ρ_z = 0.966, σ_z = 0.92, Y = 1.0, Z = 0.4677898145312322,
    α = 0.3299492385786802, r = 0.0125, δ = 0.02, εI = 4.0, εr = 0.0, κp = 0.1,
    mup = 1.015228426395939, rstar = 0.0125, φ = 1.5, φy = 0.0,
    G = 0.2, Bg = 2.8, tot_wealth = 14.0, Bh = 1.04,
    ω = 0.005, κw = 0.1, muw = 1.1, vφ = 1.713475944050497,
    frisch = 1.0, Q = 1.0, K = 10.0, N = 1.0, mc = 0.985, inv = 0.0, val = 0.0,
    pip = 0.0, nkpc = 0.0, piw = 0.0, i = 0.012499999999379603, ψp = 0.0,
    I = 0.2, div = 0.14, p = 11.2, equity = 0.0, pshare = 0.864197530864201,
    rb = 0.0075, ra = 0.0125, fisher = 0.0, tax = 0.3560606060606061,
    A = 12.96, B = 1.04, C = 0.5820937276337765, UCE = 4.434878914013179,
    CHI = 0.012706305302404835, wnkpc = 0.0, asset_mkt = 0.0, wealth = 14.0, goods_mkt = 0.0,
    zeratio=0.425, εmup=0.0, εmuw=0.0)

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
        bproduction = ta.production_blk(Hybrid)
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
        j = jacobian(b, (4,), 5, tassvals)
        Js = j.ca.Js
        # dA/drb
        @test Js[4][1][1,:] ≈ [0.01910664, -0.75874561, -0.68216018, -0.61887753, -0.56469997] atol=1e-2
        @test Js[4][1][5,:] ≈ [0.09487919, -0.47825256, -1.05770784, -1.64550569, -2.24358476] atol=1e-2
        # dB/drb
        @test Js[4][2][1,:] ≈ [0.99425083, 0.8778591 , 0.79129299, 0.71896877, 0.65654112] atol=1e-2
        @test Js[4][2][5,:] ≈ [0.84871964, 1.51917745, 2.19827569, 2.88869726, 3.59303761] atol=1e-2
        # dC/drb
        @test Js[4][3][1,:] ≈ [0.02611385, -0.12006327, -0.10602823, -0.09419434, -0.08394261] atol=1e-3
        @test Js[4][3][5,:] ≈ [0.02397805,  0.03287429,  0.04180844,  0.05085865,  0.06011889] atol=1e-3
        # dUCE/drb
        @test Js[4][4][1,:] ≈ [-0.37965504, 1.72651783, 1.53462033, 1.37684598, 1.2425804] atol=1e-1
        @test Js[4][4][5,:] ≈ [-0.32978632, -0.45494228, -0.57883471, -0.70318167, -0.8296967] atol=1e-2
        # dCHI/drb
        @test Js[4][5][1,:] ≈ [0.00052872, 0.00098339, -0.00307608, -0.00587232, -0.00787712] atol=1e-3
        @test Js[4][5][5,:] ≈ [0.00021719, -0.00206191, -0.00427481, -0.00644023, -0.00859005] atol=1e-4

        j = jacobian(b, (5,), 5, tassvals)
        # dA/dra
        @test Js[5][1][5,:] ≈ [12.55876212, 13.12224921, 13.69123629, 14.26733031, 14.8520677] atol=1e-2
        # dB/dra
        @test Js[5][2][5,:] ≈ [-0.24164763, -0.79501758, -1.35692425, -1.92922367, -2.51371582] atol=1e-2
        # dC/dra
        @test Js[5][3][5,:] ≈ [0.2380636 , 0.23233434, 0.22654506, 0.22063715, 0.220252] atol=1e-3
        # dUCE/dra
        @test Js[5][4][5,:] ≈ [-2.05280539, -2.00079259, -1.94985638, -1.89899934, -1.8285617] atol=1e-2
        # dCHI/dra
        @test Js[5][5][5,:] ≈ [0.01076234, 0.01300736, 0.015212, 0.01739816, 0.02134768] atol=1e-4
    end

    @testset "SteadyState Jacobian" begin
        Main.backwardsolver(::ta.TwoAssetHousehold) = NLsolve_anderson
        mss = ta.twoassetmodelss()
        mss.pool[1].ssargs[:backaastart] = 100
        calis = Dict{Symbol,Float64}(:Y=>1, :N=>1, :K=>10, :r=>0.0125, :rstar=>0.0125,
            :tot_wealth=>14, :δ=>0.02, :pip=>0, :κp=>0.1, :muw=>1.1, :Bh=>1.04, :Bg=>2.8,
            :G=>0.2, :eis=>0.5, :frisch=>1, :χ0=>0.25, :χ2=>2, :εI=>4, :εr=>0, :ω=>0.005,
            :κw=>0.1, :φ=>1.5, :φy=>0)
        inits = [:β=>0.976, :χ1=>6.5]
        tars = [:asset_mkt=>0, :B=>1.04]
        ss = SteadyState(mss, calis, inits, tars)
        @time solve(Hybrid, ss, ss.inits, xtol=1e-8)
        # Compare results with original Python package
        @test ss[:A] ≈ 12.96 atol=1e-5
        @test ss[:B] ≈ 1.04 atol=1e-6
        @test ss[:C] ≈ 0.5820937276337765 atol=1e-7
        @test ss[:UCE] ≈ 4.434878914013179 atol=1e-5
        @test ss[:CHI] ≈ 0.012706305302404835 atol=1e-8

        m = ta.twoassetmodel(Hybrid)
        # Directly move the household block
        m.pool[1] = mss.pool[1]
        foreach(i->steadystate!(m.pool[i], tassvals), 3:5)

        # Compute Jacobians with default epsilon from FiniteDiff.jl
        # The default epsilon is much smaller than the one set by the Python package
        @time J = TotalJacobian(m, [:rstar,:Z,:G,:w,:Y,:r], [:asset_mkt,:fisher,:wnkpc],
            tassvals, 300)
        gj = GEJacobian(J, (:rstar,:Z,:G))
        H_U = gj.H_Ublks
        # dasset_mkt/dw
        @test H_U[1,1:3] ≈ [0.01679725, 0.04170578, 0.03739954] atol=1e-3
        @test H_U[300,298:300] ≈ [-0.34252791, -0.35230779, -0.36274057] atol=1e-3
        # dasset_mkt/dY
        @test H_U[1,301:303] ≈ [-9.62962356e-1, 6.12458973e-2, 5.67827128e-2] atol=1e-3
        @test H_U[300,598:600] ≈ [-3.40122735e-1, -3.49718623e-1, -3.59957420e-1] atol=1e-3
        # dasset_mkt/dr
        @test H_U[1,601:603] ≈ [-4.16475490e-2, -1.03942468, -9.72823634e-1] atol=1e-2
        @test H_U[300,898:900] ≈ [-1.32624290e1, -1.33262797e1, -1.33873877e1] atol=1e-2
        # dfisher/dw
        @test H_U[301,1:3] ≈ [-1.51107955e-1, -1.46551915e-1, -1.42230528e-1] atol=1e-7
        @test H_U[600,298:300] ≈ [-4.51897529e-3, 2.19083483e-1, 6.52185813e-2] atol=1e-7
        # dfisher/dY
        @test H_U[301,301:303] ≈ [-4.91100852e-2, -4.58536360e-2, -4.28131192e-2] atol=1e-8
        @test H_U[600,598:600] ≈ [-4.45119066e-3,  6.80472306e-2,  1.80456267e-2] atol=1e-7
        # dfisher/dr
        @test H_U[301,601:603] ≈ [-1, -5.46380397e-2, -1.04978532e-1] atol=1e-6
        @test H_U[600,898:900] ≈ [9.17699596e-2, 9.70738865e-2, -9.03064239e-1] atol=1e-6
        # dwnkpc/dw
        @test H_U[601,1:3] ≈ [-3.38779477, 1.49386238, 1.21228316e-2] atol=1e-3
        @test H_U[900,298:300] ≈ [4.40888876e-03,  1.52001088e+00, -1.92003948] atol=2
        # dwnkpc/dY
        @test H_U[601,301:303] ≈ [2.28710864e-1,  2.16437418e-2,  1.91089705e-2] atol=1e-3
        @test H_U[900,598:600] ≈ [4.36845379e-3,  4.81102885e-3,  2.11259782e-1] atol=1e-3
        # dwnkpc/dr
        @test H_U[601,601:603] ≈ [6.91972618e-1, -1.72516350e-1, -1.57461667e-1] atol=1e-2
        @test H_U[900,898:900] ≈ [1.56318555e-1,  1.58314638e-1,  8.90424063e-1] atol=1e-2

        Gs = GMaps(gj, [:Y, :w, :r])
        Gyr = zeros(300, 300)
        Gs(Gyr, :rstar, :Y)
        @test Gyr[1,1:3] ≈ [-1.12544615, -9.42169946e-1, -6.81522845e-1] atol=1e-3
        @test Gyr[300,298:300] ≈ [-3.17242939e-2, -4.88707368e-2, 0] atol=1e-2
        Gwg = zeros(300, 300)
        Gs(Gwg, :G, :w)
        @test Gwg[1,1:3] ≈ [1.42252020e-1, 3.48736221e-2, -3.10755825e-2] atol=1e-3
        @test Gwg[300,298:300] ≈ [1.87887110e-1, 3.06551227e-1, 4.89042824e-1] atol=1
        Grz = zeros(300, 300)
        Gs(Grz, :Z, :r)
        @test Grz[1,1:3] ≈ [3.28290545e-1, 2.89395288e-1, 2.21653282e-1] atol=1e-3
        @test Grz[300,298:300] ≈ [-5.79345415e-2, -4.79892126e-1, -2.01655494e-1] atol=1e-1

        # Recompute the Jacobians with the same epsilon used by the Python package
        m.pool[1].diffargs[] = (twosided=true, epsilon=1e-4)
        J = TotalJacobian(m, [:rstar,:Z,:G,:r,:w,:Y], [:asset_mkt,:fisher,:wnkpc],
            tassvals, 300)
        @test m.pool[1].jacargs[:jacca].epsilon == 1e-4
        gj = GEJacobian(J, (:rstar,:Z,:G))

        Gs = GMaps(gj, (:Y, :w, :r))
        Gs(Gyr, :rstar, :Y)
        @test Gyr[1,1:3] ≈ [-1.12544615, -9.42169946e-1, -6.81522845e-1] atol=1e-3
        @test Gyr[300,298:300] ≈ [-3.17242939e-2, -4.88707368e-2, 0] atol=1e-2
        Gs(Gwg, :G, :w)
        @test Gwg[1,1:3] ≈ [1.42252020e-1, 3.48736221e-2, -3.10755825e-2] atol=1e-3
        @test Gwg[300,298:300] ≈ [1.87887110e-1, 3.06551227e-1, 4.89042824e-1] atol=1
        Gs(Grz, :Z, :r)
        @test Grz[1,1:3] ≈ [3.28290545e-1, 2.89395288e-1, 2.21653282e-1] atol=1e-3
        @test Grz[300,298:300] ≈ [-5.79345415e-2, -4.79892126e-1, -2.01655494e-1] atol=1e-1
    end

    @testset "Bayesian" begin
        m = ta.twoassetmodel(Hybrid)
        # Directly feed in steady state values
        foreach(x->steadystate!(x, tassvals), m.pool[isa.(m.pool, AbstractBlock)])
        exos = [:Z, :rstar, :G, :β, :εr, :εmup, :εmuw]
        endosrcs = [:r, :w, :Y]
        endos = [:Y, :pip, :i, :C, :N, :I, :w]
        tars = [:asset_mkt,:fisher,:wnkpc]
        @time J = TotalJacobian(m, vcat(exos, endosrcs), tars, tassvals, 300)
        @test J.nsrcbytar == [9, 7, 10]
        @test J.ntarbysrc == [3, 3, 2, 2, 3, 3, 1, 3, 3, 3]
        gj = GEJacobian(J, exos)
        gs = GMaps(gj, endos)

        # Compare results with Python paper replication
        G = Matrix{Float64}(undef, 300, 300)
        gs(G, :Z, :Y)
        @test G[1,1:3] ≈ [5.71674683e-1, 6.33178183e-1, 6.74896247e-1] atol=1e-2
        gs(G, :rstar, :pip)
        @test G[1,1:3] ≈ [5.17348050e-3, -9.62392510e-2, -2.03259563e-1] atol=1e-2
        gs(G, :G, :i)
        @test G[1,1:3] ≈ [1.39225068e-1, 1.53676852e-1, 1.33417204e-1] atol=1e-2
        gs(G, :β, :C)
        @test G[1,1:3] ≈ [-2.73806271e-1, -2.42176864e-1, -2.02964348e-1] atol=1e-2
        gs(G, :εr, :N)
        @test G[1,1:3] ≈ [-1.01877542, -8.19633093e-1, -5.90870311e-1] atol=1e-2
        gs(G, :εmup, :I)
        @test G[1,1:3] ≈ [-9.27050284e-1, -1.00336428, -9.84425978e-1] atol=1e-2
        gs(G, :εmuw, :w)
        @test G[1,1:3] ≈ [3.35427289e-1, 1.20935166e-1, -9.72890192e-3] atol=1e-3

        data = exampledata(:bayes)
        cols = [:y, :pi, :i, :c, :n, :I, :w]
        obs = endos.=>cols
        for (v, n) in view(obs, 1:6)
            if v in (:pip, :i)
                data[:,n] ./= 4
            else
                data[:,n] .*= 0.25 .* tassvals[v] ./ tassvals.Y
            end
        end
        data[:,:w] .*= tassvals.w ./ tassvals.Y

        shocks = [ar1shock(Symbol.((:σ, :ar), n)..., n) for n in exos]
        priors = [p for p in Iterators.flatten(
            (Symbol(:σ,n)=>InverseGamma(2.01, 0.4*(2.01-1)),
            Symbol(:ar,n)=>Beta(2.625, 2.625)) for n in exos)]
        bm = bayesian(gs, shocks, obs, priors, data)
        θ0 = append!(fill(0.4, 7), fill(0.6, 7))
        lpost = bm(θ0)
        lpri = logprior(bm, θ0)
        # Compare results with Python paper replication
        # log prior is not comparable as the Python code does not add constant terms
        @test lpost - lpri - 7*nrow(data)*log(2*pi)/2 ≈ -666.2712191687283 atol=2
        ubs = append!(fill(10.0, 7), fill(0.98, 7))
        @time θmode, rx, niter, r = mode(bm, :LD_LBFGS, θ0, lower_bounds=0.01,
            upper_bounds=ubs)
        rr = [0.07181506, 0.94403318, 0.43654337, 0.50281652, 0.09260166,
            0.94105774, 0.17386747, 0.7788629 , 0.14209558, 0.82964685,
            0.09071691, 0.90427882, 0.37310296, 0.87472214]
        @test rx[1:7] ≈ rr[1:2:13] atol=1e-2
        @test rx[8:14] ≈ rr[2:2:14] atol=1e-2

        priors2 = vcat(priors, [:φ=>Gamma(36, 1/24), :φy=>Gamma(4, 1/8),
            :κp=>Gamma(4, 1/40), :κw=>Gamma(4, 1/40), :εI=>Gamma(4, 1)])
        θ02 = vcat(θ0, Float64[tassvals[n] for n in (:φ, :φy, :κp, :κw, :εI)])
        @test_throws ArgumentError bayesian(gs, shocks, obs, priors2, data)

        J2 = TotalJacobian(m, vcat(exos, endosrcs), tars, tassvals, 300,
            dZs=[n=>ones(300) for n in exos])
        gj2 = GEJacobian(J2, exos)
        gs2 = GMaps(gj2, endos)
        bm2 = bayesian(gs2, shocks, obs, priors2, data)
        # Verify that shared ins are really shared
        for (ii, ms) in bm2.impulseupdate.plan.mmaps
            for j in 2:length(ms)
                for i in 1:length(ms[j].ins)
                    @test ms[j].ins[i] === ms[1].ins[i]
                end
            end
        end
        θ02[end-3] = 0
        lpost2 = bm2(θ02)
        lpri2 = logprior(bm2, θ02)
        @test gj2.H_Ublks ≈ gj.H_Ublks atol=1e-5
        # Verify that shared ins are really shared
        for ms in gs2.mmaps
            for j in 2:length(ms)
                for i in 1:length(ms[j].ins)
                    @test ms[j].ins[i] === ms[1].ins[i]
                end
            end
        end
        # Verify that how dZs is initially specified doesn't affect results
        dZ = zeros(300)
        ar1impulse!(dZ, 0.6)
        J3 = TotalJacobian(m, vcat(exos, endosrcs), tars, tassvals, 300,
            dZs=[n=>copy(dZ) for n in exos])
        gj3 = GEJacobian(J3, exos)
        gs3 = GMaps(gj3, endos)
        G2 = zeros(300)
        G3 = zeros(300)
        for z in exos
            for u in endos
                @test gs2(G2, z, u) ≈ gs3(G3, z, u) atol=1e-8
                @test G2 ≈ gs(G, z, u) * dZ atol=1e-8
            end
        end

        θ02[end-3] = 0.001 # 0 is not in the support of prior
        lpost2 = bm2(θ02)
        lpri2 = logprior(bm2, θ02)
        # Compare results with Python paper replication
        @test lpost2 - lpri2 - 7*nrow(data)*log(2*pi)/2 ≈ -665.7906749400908 atol=2
        # The optimization results is 
        lbs2 = append!(fill(0.01, 14), [1, 1e-3, 1e-4, 1e-4, 1e-3])
        ubs2 = append!(fill(10.0, 7), fill(0.98, 7), [3, 4, 0.7, 0.7, 5])
        # Compare results with Python paper replication
        rr2 = [0.07128583, 0.97027258, 0.46674503, 0.29153905, 0.09322817,
            0.97051952, 0.08940836, 0.86669524, 0.65455887, 0.84371779,
            0.05928499, 0.8878304, 0.14237459, 0.6476337, 1.20311558,
            0.08578049, 0.03534221, 0.00851899, 0.26707541]
        # Objective function is somewhat flat and results are not very sharp
        θ2 = vcat(rr2[1:2:13], rr2[2:2:14], rr2[15:end])
        @time θmode2, rx2, niter2, r2 = mode(bm2, :LD_LBFGS, θ2, lower_bounds=lbs2,
            upper_bounds=ubs2, verbose=true, ftol_rel=1e-6, maxeval=20)
        @test rx2[1:7] ≈ rr2[1:2:13] atol=1e-3
        @test rx2[8:14] ≈ rr2[2:2:14] atol=1e-3
        @test rx2[15:end] ≈ rr2[15:end] atol=1e-3
    end
end
