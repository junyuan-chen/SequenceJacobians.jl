@testset "Jacobian GEJacobian" begin
    @testset "RBC" begin
        using SequenceJacobians.RBC
        m = model(rbcblocks())
        calis = [:L=>1, :r=>0.01, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
        tars = [:goods_mkt=>0, :r=>0.01, :euler=>0, :Y=>1]
        inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
        ss = SteadyState(m, calis, inits, tars)
        f!(y,x) = residuals!(y, ss, x)
        solve!(GSL_Hybrids, ss, xtol=1e-10)
        J = TotalJacobian(m, [:Z,:K,:L], [:euler, :goods_mkt], ss[], 300, excluded=(:walras,))

        JK = J[:K]
        @test J[:K,:w](3)[2,1] ≈ 0.03115 atol=1e-8
        @test JK[:I](3)[:,1] ≈ [1, -0.975, 0] atol=1e-8
        @test JK[:goods_mkt](3)[:,2] ≈ [0, -1, 0.97775] atol=1e-8
        @test JK[:euler](3)[2,:] ≈ [-0.037984496124029744, 0.04863451461424746, 0] atol=1e-8
        @test JK[:Y](3)[:,1] ≈ [0, 0.035, 0] atol=1e-8
        @test JK[:r](3)[:,1] ≈ [0, -0.009911363636363545, 0] atol=1e-8
        @test JK[:C](3)[:,1] ≈ [0, 0.03225, 0] atol=1e-8

        @test J.nsrcbytar == [3, 3]
        @test J.ntarbysrc == [2, 2, 2]
        @test inlength(J) == 3
        @test tarlength(J) == 2

        @test sprint(show, J) == "TotalJacobian(euler, goods_mkt)"
        @test sprint(show, MIME("text/plain"), J) == """
            TotalJacobian{Float64} with 3 blocks, 10 variables and 300 periods:
              sources: K, L, Z
              targets: euler, goods_mkt"""

        gj = GEJacobian(J, :Z)
        # Compare results with original Python package
        @test gj.H_Ublks[1,1] ≈ 0.04863451461425212 atol=1e-7
        @test gj.H_Ublks[2,1] ≈ -0.03798449612403101 atol=1e-7
        @test gj.H_Ublks[301,1] ≈ -1
        @test gj.H_Ublks[302,1] ≈ 0.97775 atol=1e-7
        @test gj.H_Ublks[1,301] ≈ 1.2046511627906977 atol=1e-7
        @test gj.H_Ublks[1,302] ≈ -1.2381226494742499 atol=1e-7
        @test gj.H_Ublks[600,600] ≈ 1.912785714285714 atol=1e-7

        @test sprint(show, gj) == "GEJacobian(euler, goods_mkt)"
        @test sprint(show, MIME("text/plain"), gj) == """
            GEJacobian{Float64} with 300 periods:
              exogenous:  Z
              endogenous: K, L
              targets:    euler, goods_mkt"""

        G = GMaps(gj)[:Z, :C]
        # Compare results with original Python package
        @test G[1,1] ≈ 0.15969857749115557 atol=1e-7
        @test G[2,1] ≈ 0.14155046602609384 atol=1e-7
        @test G[300,300] ≈ 0.07205550732570687 atol=1e-7
    end

    @testset "KrusellSmith" begin
        using SequenceJacobians.KrusellSmith
        m = model(ksblocks())
        calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
        tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
        inits = [:β=>0.98, :Z=>0.85, :K=>3]
        ss =  SteadyState(m, calis, inits, tars)
        solve!(GSL_Hybrids, ss, xtol=1e-10)
        J = TotalJacobian(m, [:Z,:K], [:asset_mkt], ss[], 300, excluded=(:goods_mkt,))
        gj = GEJacobian(J, :Z)
        gs = GMaps(gj)
        G = gs(Matrix{Float64}(undef, 300, 300), :Z, :C)
        # Compare results with original Python package
        # Need to specify twosided=True in the Python package
        @test G[1,1:3] ≈ [2.10710661e-1, 6.58615809e-2, 5.77240526e-2] atol=1e-6
        @test G[300,298:300] ≈ [3.71126268e-2, 4.08907393e-2, 1.44162837e-1] atol=1e-6

        G = gs(G, :Z, :K)
        @test G[1,1:3] ≈ [9.23531301e-1, -6.58615809e-2, -5.77240526e-2] atol=1e-6
        @test G[300,298:300] ≈ [7.63355815e-1,  8.39331787e-1,  9.22957992e-1] atol=1e-5

        @test sprint(show, gs[:Z][:w]) == "ShiftMap{Float64}(2)"
        @test sprint(show, MIME("text/plain"), gs[:Z][:w]) == """
            ShiftMap{Float64} with 2 components:
              CompositeShift{Float64, Float64}([(-1, 0), (0, 0)], [0.031149996275251567, 0.0], (1, 1))
              CompositeShift{Float64, Float64}([(0, 0)], [1.0094753354787827], (1, 1))"""
        @test sprint(show, gs[:Z][:K]) == "MatrixMap{Float64}(1)"
        @test sprint(show, MIME("text/plain"), gs[:Z][:K])[1:58] == """
            MatrixMap{Float64} combined from 1 component:
              [0.923531 """
    end
end
