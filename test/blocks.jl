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

    @test jacobian(bhh, 1, 5, varvals) ≈ [0, 1]
    @test jacobian(bhh, 2, 5, varvals) ≈ [0, -0.975]
    @test jacobian(bhh, 8, 5, varvals) ≈ [0, 2]
end

@testset "HetBlock" begin
    using SequenceJacobians.KrusellSmith
    ins = [:r, :w, :β, :eis]
    outs = [:A, :C]
    b = kshhblock(0, 200, 500, 0.966, 0.5, 7)
    @test inputs(b) == ins
    @test outputs(b) == outs
    @test !hascache(b)
    @test nouts(b) == 2

    varvals = Dict{Symbol,Float64}(:r=>0.01, :w=>0.89, :β=>0.98, :eis=>1)
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

    # Feed in steady-state values from Python package for comparing results
    varvals = Dict{Symbol,Float64}(:r=>0.01, :w=>0.89, :β=>0.981952788061795, :eis=>1)
    b = kshhblock(0, 200, 500, 0.966, 0.5, 7)
    @test_throws ErrorException jacobian(b, 1, 5, varvals)
    steadystate!(b, varvals)
    # Check jacobian for effect on impact
    j = jacobian(b, 1, 1, varvals)
    dv = j.df[:,:,1]
    # Derivatives from Python package are based on a fixed epsilon
    # Need to specify twosided=True for better accuracy
    @test dv[1:3,1] ≈ [4.32936187, 4.20445547, 4.08329971] atol=1e-7
    @test dv[498:500,7] ≈ [0.06198027, 0.06095875, 0.05994965] atol=1e-7
    dev = j.dEVs[1]
    @test dev[1:3,1] ≈ [4.18343962, 4.06543792, 3.95090303] atol=1e-7
    @test dev[498:500,7] ≈ [0.06166387, 0.06064379, 0.05963622] atol=1e-7

    dD = j.dDs[1][:,:,1]
    @test dD[1:3,1] ≈ [-1.22588821e-4, -2.04934264e-5, -5.59833386e-5] atol=1e-9
    @test j.dYs[1] ≈ [3.047070890160419 0.09578625552963749] atol=1e-7

    # Check jacobian for 1-period ahead anticipation effect
    j = jacobian(b, 1, 2, varvals)
    dv = j.df[:,:,1]
    @test dv[1:3,1] ≈ [0, 0, 0]
    @test dv[1:3,5] ≈ [0.89735076, 0.89719971, 0.89689996] atol=1e-7
    @test dv[498:500,7,1] ≈ [0.06039384, 0.05940069, 0.05841955] atol=1e-7
    dev = j.dEVs[1]
    @test dev[1:3,1] ≈ [1.01749209e-4, 1.02397696e-4, 1.02879336e-4] atol=1e-10
    dD = j.dDs[1][:,:,2]
    @test dD[1:3,1] ≈ [-2.45911683e-3,  6.72083192e-4, -3.48413238e-5] atol=1e-10
    @test j.dYs[1][2,:] ≈ [0.6818556801588316, -0.6818556801588341] atol=1e-6

    j = jacobian(b, 1, 3, varvals)
    @test j.Es[1][1:3,1,1] ≈ [-29.52928609, -29.52928582, -29.52928554] atol=1e-7
    @test j.Es[1][498:500,7,2] ≈ [163.33189788, 165.91565888, 168.53432082] atol=1e-7
    @test j.Es[2][1:3,1,1] ≈ [-1.39279428, -1.38938866, -1.38593711] atol=1e-7
    @test j.Es[2][498:500,7,2] ≈ [3.89850409, 3.9474403, 3.99701225] atol=1e-7

    j = jacobian(b, 1, 5, varvals)
    @test j.Js[1][1][1,:] ≈ [3.04707089, 0.68185568, 0.64125217, 0.60439044, 0.57061299] atol=1e-6
    @test j.Js[1][1][5,:] ≈ [2.79839241, 3.42915491, 4.05731394, 4.68424741, 5.31162232] atol=1e-6
    @test j.Js[1][2][1,:] ≈ [0.09578626, -0.68185568, -0.64125217, -0.60439044, -0.57061299] atol=1e-6
    @test j.Js[1][2][5,:] ≈ [0.08926242, 0.12116544, 0.15313027, 0.18541724, 0.21771175] atol=1e-6
end
