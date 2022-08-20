@testset "autocov autocor" begin
    using SequenceJacobians.KrusellSmith
    m = model(ksblocks())
    calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
    tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
    inits = [:β=>0.98, :Z=>0.85, :K=>3]
    ss =  SteadyState(m, calis, inits, tars)
    solve!(GSL_Hybrids, ss, xtol=1e-10)
    J = TotalJacobian(m, [:Z,:K], [:asset_mkt], getvarvals(ss), 300, excluded=(:goods_mkt,))
    GJ = GEJacobian(J, :Z, keepH_U=true)

    dZ1 = 0.9.^(0:299)
    dY1 = getG!(GJ, :Z, :Y) * dZ1
    dC1 = getG!(GJ, :Z, :C) * dZ1
    dK1 = getG!(GJ, :Z, :K) * dZ1
    dZ2 = zeros(300)
    dZ2[1] = 1
    dY2 = view(getG!(GJ, :Z, :Y), :,1)
    dC2 = view(getG!(GJ, :Z, :C), :,1)
    dK2 = view(getG!(GJ, :Z, :K), :,1)
    dX = vcat(dZ1, dY1, dC1, dK1, dZ2, dY2, dC2, dK2)
    Σ = autocov(reshape(dX, 300, 4, 2), [0.1, 0.2])
    @test Σ[1] ≈ 0.09263157894736844 atol=1e-8
    @test Σ[100,2,2] ≈ 1.1467655790671985e-5 atol=1e-10
    @test Σ[150,2,3] ≈ -3.184957224656828e-7 atol=1e-12
    @test Σ[200,2,2] ≈ -2.4343217950007922e-8 atol=1e-12
    @test Σ[200,3,4] ≈ -4.5565272153303164e-7 atol=1e-12

    corr = autocor(reshape(dX, 300, 4, 2), [0.1, 0.2])
    @test corr[1,1,:] ≈ [1.0, 0.99469738, 0.78762659, 0.60438467] atol=1e-6
    @test corr[2,4,:] ≈ [0.45349869, 0.54275537, 0.88132246, 0.98355236] atol=1e-6
end

