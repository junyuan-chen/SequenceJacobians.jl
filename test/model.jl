@testset "SequenceSpaceModel" begin
    using SequenceJacobians.RBC
    bfirm, bhh, bmkt, bss = rbcblocks()
    m = model([bfirm, bhh, bmkt, bss])
end

@testset "SteadyState" begin
    @testset "RBC" begin
        using SequenceJacobians.RBC
        bfirm, bhh, bmkt, bss = rbcblocks()
        m = model([bfirm, bhh, bmkt, bss])
        calis = [:L=>1, :r=>0.01, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
        tars = [:goods_mkt=>0, :rss=>0.01, :euler=>0, :Yss=>1]
        inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
        ss = SteadyState(m, calis, tars, inits)
        f(x) = criterion!(ss, x)
        f!(y,x) = residuals!(y, ss, x)
        r = solve!(GSL_Hybrids, f!, ss.inits, xtol=1e-10)
        # Compare results with original Python package
        @test r[1] ≈ [0.9900990099009883, 0.9658914728682162, 0.8816460975214576,
            3.1428571428570864] atol=1e-8
        @test ss.varvals[:φ] ≈ 0.9658914728682162 atol=1e-8
        @test ss.varvals[:β] ≈ 0.9900990099009883 atol=1e-8
        @test ss.varvals[:K] ≈ 3.1428571428570864 atol=1e-8
        @test ss.varvals[:Z] ≈ 0.8816460975214576 atol=1e-8

        @test_throws ArgumentError solve!(Roots_Default_Solver, ss)
    end

    @testset "KrusellSmith" begin
        using SequenceJacobians.KrusellSmith
        bhh, bfirm, bmkt, bss = ksblocks()
        m = model([bhh, bfirm, bmkt, bss])
        calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
        tars = [:rss=>0.01, :Yss=>1, :asset_mkt=>0]
        inits = [:β=>0.98, :Z=>0.85, :K=>3]
        ss =  SteadyState(m, calis, tars, inits)
        # NLsolve reevaluates Jacobians too many times
        solve!(GSL_Hybrids, ss, xtol=1e-10)
        # Compare results with original Python package
        @test ss.varvals[:β] ≈ 0.981952788061795 atol=1e-8
        @test ss.varvals[:Z] ≈ 0.8816460975214567 atol=1e-8
        @test ss.varvals[:K] ≈ 3.142857142857143 atol=1e-8
    end
end
