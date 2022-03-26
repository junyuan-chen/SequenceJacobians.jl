@testset "SimpleBlock" begin
    using SequenceJacobians.RBC
    firm, household, mkt_clearing = RBC.firm, RBC.household, RBC.mkt_clearing
    ins = [:K, :L, :Z, :α, :δ]
    outs = [:r, :w, :Y]
    b = block(firm, [lag(:K), :L, :Z, :α, :δ], outs)
    @test b isa SimpleBlock
    @test inputs(b) == ins
    @test ssinputs(b) == Set(ins)
    @test outputs(b) == outs
    @test b(10.0, 1.0, 1.0, 0.3, 0.05) == firm(10.0, 1.0, 1.0, 0.3, 0.05)

    @test_throws MethodError block(firm, (), outs)
    @test_throws ArgumentError block(firm, ins, ())
    @test_throws ArgumentError block(firm, ins, outs, ssins=:n)
    @test_throws ArgumentError block(firm, ins, ins)

    ins = [:K, :K, :L, :w, :eis, :frisch, :φ, :δ]
    outs = [:C, :I]
    b = block(household, [:K, lag(:K), :L, :w, :eis, :frisch, :φ, :δ], outs)
    @test inputs(b) == ins

    ins = [:r, :r, :C, :C, :Y, :I, :K, :K, :L, :w, :eis, :β]
    outs = [:goods_mkt, :euler, :walras]
    b = block(mkt_clearing,
        [:r, lead(:r), :C, lead(:C), :Y, :I, :K, lag(:K), :L, :w, :eis, :β], outs)
    @test inputs(b) == ins

    bfirm, bhh, bmkt, bss = rbcblocks()
    varvals = Dict{Symbol,Float64}(:K=>2,:L=>1,:w=>1,:eis=>1,:frisch=>1,:φ=>0.9,:δ=>0.025)
    steadystate!(bhh, varvals)
    @test varvals[:C] ≈ 1.1111111111111112
    @test varvals[:I] ≈ 0.05

    @test jacobian(bhh, 1, varvals) ≈ [0, 1]
    @test jacobian(bhh, 2, varvals) ≈ [0, -0.975]
    @test jacobian(bhh, 8, varvals) ≈ [0, 2]
end

@testset "HetBlock" begin
    using SequenceJacobians.KrusellSmith
    ins = [:r, :w, :β, :eis]
    outs = [:A, :C]
    ca = Dict{Symbol,Any}()
    b = kshhblock(0, 200, 500, 0.966, 0.5, 7)
    @test inputs(b) == ins
    @test outputs(b) == outs
    @test !hascache(b)
    @test nouts(b) == 2

    varvals = Dict{Symbol,Any}(:r=>0.01, :w=>0.89, :β=>0.98, :eis=>1)
    steadystate!(b, varvals)
    # Compare results with original Python package
    a = b.ha.a
    @test all(a[1:4,1] .== 0)
    @test all(a[1:3,3] .== 0)
    @test a[5:8,1] ≈ [1.66414395e-3, 3.43258549e-3, 5.22507103e-3, 7.04179415e-3] atol=1e-7
    @test a[1:4,7] ≈ [0.90054142, 0.90383955, 0.90718218, 0.9105699] atol=1e-7
    @test a[end-3:end,7] ≈ [190.6133331, 193.1797762, 195.78086653, 198.41707144] atol=1e-6
    D = b.ha.D
    @test D[1:3,1] ≈ [1.41373523e-2, 4.08282500e-5, 3.83039718e-5] atol=1e-8
    @test D[1:3,4] ≈ [1.42736359e-2, 1.82294210e-2, 2.57256128e-2] atol=1e-8
    @test D[1:3,7] ≈ [3.38553410e-8, 9.33874325e-8, 1.32361967e-7] atol=1e-8
    @test varvals[:C] ≈ 0.9112915134243005 atol=1e-7
    @test varvals[:A] ≈ 2.1291511229699926 atol=1e-7
end
