@testset "SimpleBlock" begin
    using SequenceJacobians.RBC
    firm, household, mkt_clearing = RBC.firm, RBC.household, RBC.mkt_clearing
    ins = [:K, :L, :Z, :α, :δ]
    outs = (:r, :w, :Y)
    b = block(firm, [lag(:K), :L, :Z, :α, :δ], outs)
    @test b isa SimpleBlock
    @test inputs(b) === (ins...,)
    @test ssinputs(b) == Set(ins)
    @test outputs(b) === (outs...,)
    @test b(10.0, 1.0, 1.0, 0.3, 0.05) === NamedTuple{outs}(firm(10.0, 1.0, 1.0, 0.3, 0.05))

    @test_throws ArgumentError block(firm, (), outs)
    @test_throws ArgumentError block(firm, ins, ())
    @test_throws ArgumentError block(firm, ins, outs, ssins=:n)
    @test_throws ArgumentError block(firm, ins, ins)

    ins = [:K, :K, :L, :w, :eis, :frisch, :φ, :δ]
    outs = [:C, :I]
    b = block(household, [:K, lag(:K), :L, :w, :eis, :frisch, :φ, :δ], outs)
    @test inputs(b) == (ins...,)

    ins = [:r, :r, :C, :C, :Y, :I, :K, :K, :L, :w, :eis, :β]
    outs = [:goods_mkt, :euler, :walras]
    b = block(mkt_clearing,
        [:r, lead(:r), :C, lead(:C), :Y, :I, :K, lag(:K), :L, :w, :eis, :β], outs)
    @test inputs(b) == (ins...,)

    bfirm, bhh, bmkt, bss = rbcblocks()
    varvals = (K=2, L=1, w=1, eis=1, frisch=1, φ=0.9, δ=0.025)
    varvals = steadystate!(bhh, varvals)
    @test varvals[:C] ≈ 1.1111111111111112
    @test varvals[:I] ≈ 0.05

    @test jacobian(bhh, Val(1), 5, varvals) ≈ [0, 1]
    @test jacobian(bhh, Val(2), 5, varvals) ≈ [0, -0.975]
    @test jacobian(bhh, Val(8), 5, varvals) ≈ [0, 2]
end

@testset "HetBlock" begin
    using SequenceJacobians.KrusellSmith
    ins = (:r, :w, :β, :eis)
    outs = (:A, :C)
    b = kshhblock(0, 200, 500, 0.966, 0.5, 7)
    @test inputs(b) === ins
    @test outputs(b) === outs
    @test !hascache(b)
    @test outlength(b) == 2

    varvals = (r=0.01, w=0.89, β=0.98, eis=1)
    varvals = steadystate!(b, varvals)
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
    varvals = (r=0.01, w=0.89, β=0.981952788061795, eis=1)
    b = kshhblock(0, 200, 500, 0.966, 0.5, 7)
    @test_throws ErrorException jacobian(b, Val(1), 5, varvals)
    varvals = steadystate!(b, varvals)
    # Check jacobian for effect on impact
    j = jacobian(b, Val(1), 1, varvals)
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
    j = jacobian(b, Val(1), 2, varvals)
    dv = j.df[:,:,1]
    @test dv[1:3,1] ≈ [0, 0, 0]
    @test dv[1:3,5] ≈ [0.89735076, 0.89719971, 0.89689996] atol=1e-7
    @test dv[498:500,7,1] ≈ [0.06039384, 0.05940069, 0.05841955] atol=1e-7
    dev = j.dEVs[1]
    @test dev[1:3,1] ≈ [1.01749209e-4, 1.02397696e-4, 1.02879336e-4] atol=1e-10
    dD = j.dDs[1][:,:,2]
    @test dD[1:3,1] ≈ [-2.45911683e-3,  6.72083192e-4, -3.48413238e-5] atol=1e-10
    @test j.dYs[1][2,:] ≈ [0.6818556801588316, -0.6818556801588341] atol=1e-6

    j = jacobian(b, Val(1), 3, varvals)
    @test j.Es[1][1:3,1,1] ≈ [-29.52928609, -29.52928582, -29.52928554] atol=1e-7
    @test j.Es[1][498:500,7,2] ≈ [163.33189788, 165.91565888, 168.53432082] atol=1e-7
    @test j.Es[2][1:3,1,1] ≈ [-1.39279428, -1.38938866, -1.38593711] atol=1e-7
    @test j.Es[2][498:500,7,2] ≈ [3.89850409, 3.9474403, 3.99701225] atol=1e-7

    j = jacobian(b, Val(1), 5, varvals)
    @test j.Js[1][1][1,:] ≈ [3.04707089, 0.68185568, 0.64125217, 0.60439044, 0.57061299] atol=1e-6
    @test j.Js[1][1][5,:] ≈ [2.79839241, 3.42915491, 4.05731394, 4.68424741, 5.31162232] atol=1e-6
    @test j.Js[1][2][1,:] ≈ [0.09578626, -0.68185568, -0.64125217, -0.60439044, -0.57061299] atol=1e-6
    @test j.Js[1][2][5,:] ≈ [0.08926242, 0.12116544, 0.15313027, 0.18541724, 0.21771175] atol=1e-6
end

@testset "CombinedBlock" begin
    using SequenceJacobians.TwoAsset
    pricing = SequenceJacobians.TwoAsset.pricing
    ins = (:mc, lead(:r), :Y, lead(:Y), :κp, :mup)
    outs = :pip
    bpricing = block(pricing, (:pip, lead(:pip), ins...), :nkpc)
    mpricing = model(bpricing)
    sspricing = SteadyState(mpricing, [:mc=>0.985, :r=>0.0125, :Y=>1, :κp=>0.1, :mup=>1.015228426395939], :pip=>0.1, :nkpc=>0)
    @test_throws ArgumentError block(sspricing, ins, outs, :nkpc)
    b = block(sspricing, ins, outs, :nkpc, solver=Roots_Default)
    ins = (:mc, :r, :Y, :κp, :mup)
    @test inputs(b) == ins
    @test invars(b) == ins
    @test ssinputs(b) == Set(ins)
    @test outputs(b) == (:pip,)
    @test !hascache(b)
    @test outlength(b) == 1
    @test outlength(b, 1) == 1
    varvals = getvarvals(sspricing)
    steadystate!(b, varvals)
    @test getval(b.ss, :pip) ≈ 0 atol=1e-8
    @test getval(b.ss, :nkpc) ≈ 0 atol=1e-8

    # Compare results with original Python package
    J = jacobian(b, 3, varvals)
    @test all(isapprox.(J.Gs[:κp][:pip], 0, atol=1e-8))
    Jmc = [0.1 0.09876543 0.09754611;
           0   0.1        0.09876543;
           0   0          0.1        ]
    @test J.Gs[:mc][:pip] ≈ Jmc atol=1e-8
    Jmup = [0.0970225 0.09582469 0.09464167;
            0         0.0970225  0.09582469;
            0         0          0.0970225  ]
    @test J.Gs[:mup][:pip] ≈ Jmup atol=1e-8
    @test all(isapprox.(J.Gs[:Y][:pip], 0, atol=1e-8))
    @test all(isapprox.(J.Gs[:r][:pip], 0, atol=1e-8))

    arbitrage = SequenceJacobians.TwoAsset.arbitrage
    ins = (lead(:div), lead(:r))
    outs = :p
    barbitrage = block(arbitrage, (:p, lead(:p), ins...), :equity)
    b = block(barbitrage, ins, outs, :equity, [:div=>0.14, :r=>0.0125], :p=>10, :equity=>0,
        solver=Brent(), ssargs=(:x0=>(5,15),))
    varvals = steadystate!(b, getvarvals(b.ss))
    @test getval(b.ss, :p) ≈ 11.2 atol=1e-8

    # Compare results with original Python package
    J = jacobian(b, 3, varvals)
    Jdiv = [0 0.98765432 0.97546106;
            0 0          0.98765432;
            0 0          0          ]
    @test J.Gs[:div][:p] ≈ Jdiv atol=1e-8
    Jr = [0 -11.0617284 -10.92516385;
          0 0           -11.0617284;
          0 0           0            ]
    @test J.Gs[:r][:p] ≈ Jr atol=1e-8

    labor = SequenceJacobians.TwoAsset.labor
    investment = SequenceJacobians.TwoAsset.investment
    blabor = block(labor, [:Y, :w, lag(:K), :Z, :α], [:N, :mc])
    binvest = block(investment, [:Q, lead(:Q), :K, lag(:K), lead(:K), lead(:r), lead(:N), lead(:mc), lead(:Z), :δ, :εI, :α], [:inv, :val])
    calis = [:Y, :w, :Z, :α, :r, :δ, :εI]
    b = block([blabor, binvest], [:Y, :w, :Z, :r], [:Q, :K], [:inv, :val],
        calis.=>[1.0, 0.66, 0.4677898145312322, 0.3299492385786802, 0.0125, 0.02, 4],
        [:Q=>2, :K=>11], [:inv, :val].=>0.0, solver=GSL_Hybrids)
    varvals = steadystate!(b, getvarvals(b.ss))
    @test varvals[:Q] ≈ 1 atol=1e-8
    @test varvals[:K] ≈ 10 atol=1e-8

    # Compare results with original Python package
    J = jacobian(b, 3, varvals)
    Jyk = [0 0.03789632 0.03714605;
           0 0.03761037 0.07490055;
           0 0.03746678 0.0746146  ]
    @test J.Gs[:Y][:K] ≈ Jyk atol=1e-8
    Jrq = [0 -0.97663311 -0.95729755;
           0 0.00736934  -0.97297837;
           0 0.00370042   0.00736934 ]
    @test J.Gs[:r][:Q] ≈ Jrq atol=1e-8
end

@testset "SolvedBlock" begin
    using SequenceJacobians.TwoAsset
    pricing = SequenceJacobians.TwoAsset.pricing
    ins = [:mc, lead(:r), :Y, lead(:Y), :κp, :mup]
    outs = :pip
    bpricing = block(pricing, union([:pip, lead(:pip)], ins), :nkpc)
    mpricing = model(bpricing)
    sspricing = SteadyState(mpricing, [:mc=>0.985, :r=>0.0125, :Y=>1, :κp=>0.1, :mup=>1.015228426395939], :pip=>0.1, :nkpc=>0)
    b = block(sspricing, ins, outs, :nkpc, solver=Roots_Default)
    varvals = steadystate!(b, getvarvals(b.ss))
    J = jacobian(b, 3, varvals)
    bj = block(b, J)
    @test inputs(bj) == inputs(b)
    @test invars(bj) == invars(b)
    @test ssinputs(bj) == ssinputs(b)
    @test outputs(bj) == outputs(b)
    @test hascache(bj) == hascache(b)
    @test outlength(bj) == outlength(b)
    @test outlength(bj, 1) == outlength(b, 1)

    @test_throws ErrorException steadystate!(bj, varvals)
    @test jacbyinput(bj) == false
    @test jacobian(bj, 3, varvals) === J
    @test_throws ErrorException jacobian(bj, 5, varvals)
end
