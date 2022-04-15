@testset "SequenceSpaceModel" begin
    using SequenceJacobians.RBC
    m = model(rbcblocks())

    @test sprint(show, m) == "{20, 29} SequenceSpaceModel"
    @test sprint(show, MIME("text/plain"), m) == """
        {20, 29} SequenceSpaceModel with 3 blocks and 17 variables:
          SimpleBlock(firm)
          SimpleBlock(household)
          SimpleBlock(mkt_clearing)"""
end

@testset "SteadyState" begin
    @testset "RBC" begin
        using SequenceJacobians.RBC
        m = model(rbcblocks())
        calis = [:L=>1, :r=>0.01, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
        tars = [:goods_mkt=>0, :r=>0.01, :euler=>0, :Y=>1]
        inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
        ss = SteadyState(m, calis, inits, tars)
        f(x) = criterion!(ss, x)
        f!(y,x) = residuals!(y, ss, x)
        r = solve!(GSL_Hybrids, f!, ss.inits, xtol=1e-10)
        # Compare results with original Python package
        @test r[1] ≈ [0.9900990099009883, 0.9658914728682162, 0.8816460975214576,
            3.1428571428570864] atol=1e-8
        @test getval(ss, :φ) ≈ 0.9658914728682162 atol=1e-8
        @test getval(ss, :β) ≈ 0.9900990099009883 atol=1e-8
        @test getval(ss, :K) ≈ 3.1428571428570864 atol=1e-8
        @test getval(ss, :Z) ≈ 0.8816460975214576 atol=1e-8

        @test_throws ArgumentError solve!(Roots_Default, ss)
    end

    @testset "KrusellSmith" begin
        using SequenceJacobians.KrusellSmith
        m = model(ksblocks())
        calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
        tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
        inits = [:β=>0.98, :Z=>0.85, :K=>3]
        ss = SteadyState(m, calis, inits, tars)
        @test targets(ss) == (:Y, :r, :asset_mkt)
        solve!(GSL_Hybrids, ss, xtol=1e-10)
        # Compare results with original Python package
        @test getval(ss, :β) ≈ 0.981952788061795 atol=1e-8
        @test getval(ss, :Z) ≈ 0.8816460975214567 atol=1e-8
        @test getval(ss, :K) ≈ 3.142857142857143 atol=1e-8
        # Results returned by the solver may be the guess for the next iteration
        @test ss.inits ≈ [getval(ss, n) for n in inputs(ss)] atol=1e-8

        @test sprint(show, ss) == "3×3 SteadyState{Float64}"
        @test sprint(show, MIME("text/plain"), ss) == """
            3×3 SteadyState{Float64} with 3 blocks and 14 variables:
              unknowns: Z, K, β
              targets:  Y, r, asset_mkt"""

        @testset "Anderson acceleration" begin
            m = model(ksblocks())
            bhh = m.pool[1]
            backwardsolver(::KSHousehold) = NLsolve_anderson
            bhh.ssargs[:mbackward] = 5
            ss = SteadyState(m, calis, inits, tars)
            solve!(GSL_Hybrids, ss, xtol=1e-10)
            # Compare results with original Python package
            @test getval(ss, :β) ≈ 0.981952788061795 atol=1e-8
            @test getval(ss, :Z) ≈ 0.8816460975214567 atol=1e-8
            @test getval(ss, :K) ≈ 3.142857142857143 atol=1e-8
            backwardsolver(::KSHousehold) = nothing
            forwardsolver(::KSHousehold) = NLsolve_anderson
            m = model(ksblocks())
            bhh = m.pool[1]
            bhh.ssargs[:mforward] = 5
            ss = SteadyState(m, calis, inits, tars)
            solve!(GSL_Hybrids, ss, xtol=1e-10)
            # Compare results with original Python package
            @test getval(ss, :β) ≈ 0.981952788061795 atol=1e-8
            @test getval(ss, :Z) ≈ 0.8816460975214567 atol=1e-8
            @test getval(ss, :K) ≈ 3.142857142857143 atol=1e-8
            forwardsolver(::KSHousehold) = nothing
        end
    end
end
