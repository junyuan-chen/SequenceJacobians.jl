@testset "Jacobian GEJacobian" begin
    @testset "RBC" begin
        using SequenceJacobians.RBC
        bfirm, bhh, bmkt, bss = rbcblocks()
        m = model([bfirm, bhh, bmkt, bss])
        calis = [:L=>1, :r=>0.01, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
        tars = [:goods_mkt=>0, :rss=>0.01, :euler=>0, :Yss=>1]
        inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
        ss = SteadyState(m, calis, tars, inits)
        f!(y,x) = residuals!(y, ss, x)
        r = fsolve(f!, ss.inits, tol=1e-10)
        J = TotalJacobian(m, [:Z,:K,:L], [:euler, :goods_mkt], ss.varvals, 300, (bss, :walras))

        JK = J.totals[:K]
        @test JK[:w].S.v ≈ [0.03115] atol=1e-7
        @test JK[:I].S.v[[JK[:I].S.d[(-1,0)],JK[:I].S.d[(0,0)]]] ≈ [-0.975, 1.0]
        @test JK[:goods_mkt].S.v[[JK[:goods_mkt].S.d[(-1,0)],JK[:goods_mkt].S.d[(0,0)]]] ≈ [0.97775, -1.0] atol=1e-7
        @test JK[:euler].S.v[[JK[:euler].S.d[(-1,0)],JK[:euler].S.d[(0,0)]]] ≈ [-0.037984496124029744, 0.04863451461424746] atol=1e-7
        @test JK[:Y].S.v ≈ [0.035] atol=1e-7
        @test JK[:r].S.v ≈ [-0.009911363636363545] atol=1e-7
        @test JK[:C].S.v ≈ [0.03225] atol=1e-7

        @test J.ntarsrc == [3, 3]

        GJ = GEJacobian(J, :Z, keepH_U=true)
        # Compare results with original Python package
        @test GJ.H_U[1,1] ≈ 0.04863451461425212 atol=1e-7
        @test GJ.H_U[2,1] ≈ -0.03798449612403101 atol=1e-7
        @test GJ.H_U[301,1] ≈ -1
        @test GJ.H_U[302,1] ≈ 0.97775 atol=1e-7
        @test GJ.H_U[1,301] ≈ 1.2046511627906977 atol=1e-7
        @test GJ.H_U[1,302] ≈ -1.2381226494742499 atol=1e-7
        @test GJ.H_U[600,600] ≈ 1.912785714285714 atol=1e-7

        G = getG!(GJ, :Z, :C)
        # Compare results with original Python package
        @test G[1,1] ≈ 0.15969857749115557 atol=1e-7
        @test G[2,1] ≈ 0.14155046602609384 atol=1e-7
        @test G[300,300] ≈ 0.07205550732570687 atol=1e-7 
    end
end
