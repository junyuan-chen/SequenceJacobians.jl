@testset "SequenceSpaceModel" begin
    using SequenceJacobians.RBC
    m = model(rbcblocks())

    @test SimpleDiGraph(m) === m.dag
    @test edgetype(m) == edgetype(m.dag)
    @test eltype(m) == Int
    @test nv(m) == 20
    @test ne(m) == 29
    @test vertices(m) == 1:20
    @test edges(m) == edges(m.dag)
    @test is_directed(m)
    @test has_vertex(m, 1)
    @test has_edge(m, 4, 1)
    @test inneighbors(m, 1) == 4:8
    @test outneighbors(m, 1) == 9:11
    z = zero(m)
    @test nv(z) == ne(z) == 0
    @test z isa SequenceSpaceModel

    @test srcs(m) === m.srcs
    @test vsrcs(m) == [:K, :L, :Z, :α, :δ, :eis, :frisch, :φ, :β]
    @test sssrcs(m) === m.sssrcs
    @test vsssrcs(m) == vsrcs(m)
    @test dests(m) === m.dests
    @test vdests(m) == [:goods_mkt, :euler, :walras]
    @test isblock(m, 1)
    @test !isblock(m, 4)

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
        @test varvalstype(ss) == typeof(ss.varvals[])
        @test blkstype(ss) == typeof(ss.blks)
        @test scalarinputs(ss) == (:β, :φ, :Z, :K)
        @test arrayinputs(ss) == ()
        @test scalartargets(ss) == (:Y, :r, :euler, :goods_mkt)
        @test arraytargets(ss) == ()
        @test ss[] == ss.varvals[]

        r = solve(Hybrid, ss, ss.inits, ftol=1e-10)
        # Compare results with original Python package
        @test r.x ≈ [0.9900990099009883, 0.9658914728682162, 0.8816460975214576,
            3.1428571428570864] atol=1e-8
        @test ss[:φ] ≈ 0.9658914728682162 atol=1e-8
        @test ss[:β] ≈ 0.9900990099009883 atol=1e-8
        @test ss[:K] ≈ 3.1428571428570864 atol=1e-8
        @test ss[:Z] ≈ 0.8816460975214576 atol=1e-8

        @test_throws ArgumentError _solve!(Roots_Default, ss)
    end

    @testset "KrusellSmith" begin
        using SequenceJacobians.KrusellSmith
        m = model(ksblocks())
        calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
        tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
        inits = [:β=>0.98, :Z=>0.85, :K=>3]
        ss = SteadyState(m, calis, inits, tars)
        @test targets(ss) == (:Y, :r, :asset_mkt)
        _solve!(Hybrid, ss, xtol=1e-10)
        # Compare results with original Python package
        @test ss[:β] ≈ 0.981952788061795 atol=1e-8
        @test ss[:Z] ≈ 0.8816460975214567 atol=1e-8
        @test ss[:K] ≈ 3.142857142857143 atol=1e-8
        # Results returned by the solver may be the guess for the next iteration
        @test ss.inits ≈ [ss[n] for n in inputs(ss)] atol=1e-8

        @test sprint(show, ss) == "3×3 SteadyState{Float64}"
        @test sprint(show, MIME("text/plain"), ss) == """
            3×3 SteadyState{Float64} with 3 blocks and 14 variables:
              unknowns: Z, K, β
              targets:  Y, r, asset_mkt"""

        @testset "Anderson acceleration" begin
            Main.backwardsolver(::KSHousehold) = NLsolve_anderson
            m = model(ksblocks())
            bhh = m.pool[1]
            ss = SteadyState(m, calis, inits, tars)
            solve(Hybrid, ss, ss.inits, ftol=1e-10)
            # Compare results with original Python package
            @test ss[:β] ≈ 0.981952788061795 atol=1e-8
            @test ss[:Z] ≈ 0.8816460975214567 atol=1e-8
            @test ss[:K] ≈ 3.142857142857143 atol=1e-8
            Main.backwardsolver(::KSHousehold) = nothing
            Main.forwardsolver(::KSHousehold) = NLsolve_anderson
            m = model(ksblocks())
            bhh = m.pool[1]
            bhh.ssargs[:mforward] = 10
            ss = SteadyState(m, calis, inits, tars)
            solve(Hybrid, ss, ss.inits, xtol=1e-8)
            # Compare results with original Python package
            @test ss[:β] ≈ 0.981952788061795 atol=1e-8
            @test ss[:Z] ≈ 0.8816460975214567 atol=1e-8
            @test ss[:K] ≈ 3.142857142857143 atol=1e-7
            Main.forwardsolver(::KSHousehold) = nothing
        end
    end
end
