@testset "KrusellSmith" begin
    using SequenceJacobians.KrusellSmith
    m = model(ksblocks())
    calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
    tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
    inits = [:β=>0.98, :Z=>0.85, :K=>3]
    ss =  SteadyState(m, calis, inits, tars)
    solve(Hybrid, ss, ss.inits, ftol=1e-10)
    J = TotalJacobian(m, [:Z,:K], [:asset_mkt], ss[], 300, excluded=(:goods_mkt,))
    gj = GEJacobian(J, :Z)
    gs = GMaps(gj)

    @testset "allcov allcor correlogram" begin
        dZ1 = 0.9.^(0:299)
        dY1 = gs(:Z,:Y) * dZ1
        dC1 = gs(:Z,:C) * dZ1
        dK1 = gs(:Z,:K) * dZ1
        dZ2 = zeros(300)
        dZ2[1] = 1
        dY2 = gs(:Z,:Y) * dZ2
        dC2 = gs(:Z,:C) * dZ2
        dK2 = gs(:Z,:K) * dZ2
        dX = reshape(vcat(dZ1, dY1, dC1, dK1, dZ2, dY2, dC2, dK2), 300, 4, 2)
        σ = [0.1, 0.2]
        Σ = allcov(dX, σ)
        @test Σ[1] ≈ sum(dZ1.^2)*σ[1]^2 + sum(dZ2.^2)*σ[2]^2
        @test Σ[1] ≈ 0.09263157894736844 atol=1e-8
        @test Σ[100,2,2] ≈ 1.1467655790671985e-5 atol=1e-10
        @test Σ[150,2,3] ≈ -3.184957224656828e-7 atol=1e-12
        @test Σ[200,2,2] ≈ -2.4343217950007922e-8 atol=1e-12
        @test Σ[200,3,4] ≈ -4.5565272153303164e-7 atol=1e-12
        Σ1 = allcov(dX)
        @test Σ1[1] ≈ sum(dZ1.^2) + sum(dZ2.^2)
        ca = FFTWAllCovCache(300, 4, 2)
        out = similar(Σ)
        allcov!(out, ca, dX, σ)
        @test out ≈ Σ
        allcov!(out, ca, dX)
        @test out ≈ Σ1
        @test sprint(show, ca) == "FFTWAllCovCache"

        corr = allcor(dX, σ)
        @test corr[1,1,:] ≈ [1.0, 0.99469738, 0.78762659, 0.60438467] atol=1e-6
        @test corr[2,4,:] ≈ [0.45349869, 0.54275537, 0.88132246, 0.98355236] atol=1e-6
        corr1 = allcor(dX)
        @test corr1[2,2,1] ≈ Σ1[2,2,1] / (sqrt(Σ1[1,1,1])*sqrt(Σ1[1,2,2]))
        out = similar(corr)
        allcor!(out, ca, dX, σ)
        @test out ≈ corr
        allcor!(out, ca, dX)
        @test out ≈ corr1

        r = correlogram(dX, (1:4).=>1, lagmin=-50, lagmax=50, σ=σ)
        @test size(r) == (101, 4)
        @test r[1,:] == corr[51,:,1]
        @test r[end,:] == corr[51,1,:]
        out = similar(r)
        ca = FFTWAllCovCache(300, 4, 2)
        correlogram!(out, ca, dX, (1:4).=>1, lagmin=-50, lagmax=50, σ=σ)
        @test out ≈ r

        r = correlogram(dX, 2=>1, lagmin=-50, lagmax=50)
        @test size(r) == (101, 1)
        r = correlogram(dX, 2=>4, lagmin=10, lagmax=20, σ=σ)
        @test view(r, :) ≈ corr[11:21,4,2]
        r = correlogram(dX, 2=>2, lagmin=-10, lagmax=-1, σ=σ)
        @test view(r, :) ≈ corr[11:-1:2,2,2]
        r = correlogram(dX, 2=>2)
        @test length(r) == 1
        @test r[1] ≈ 1
        @test_throws ArgumentError correlogram(dX, 2=>1, lagmin=5, lagmax=0)
        @test_throws ArgumentError correlogram(dX, 2=>1, lagmin=0, lagmax=300)
        @test_throws ArgumentError correlogram(dX, 2=>1, lagmin=-300, lagmax=0)
    end

    @testset "simulate" begin
        ε = randn(399)
        gs = GMaps(gj)
        s = simulate(gs, :Z, :K, ε, ARMAProcess(0.9, ()))
        @test size(s) == (100, 1)

        ε = randn(100)
        @test_throws ArgumentError simulate(gs, :Z, :K, ε, ARMAProcess(0.9, ()))
    end
end

