using SequenceJacobians.RBC
function rbcss()
    bfirm, bhh, bmkt, bss = rbcblocks()
    m = model([bfirm, bhh, bmkt])
    calis = [:L=>1, :r=>0.01, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
    tars = [:goods_mkt=>0, :r=>0.01, :euler=>0, :Y=>1]
    inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
    return SteadyState(m, calis, inits, tars)
end

@testset "GSL" begin
    n = 5
    roots = collect(1.0:5.0)
    function f!(out, x)
        out .= (x .- roots).^2
    end
    x0 = Float64[1.0, 5.0, 2.0, 1.5, -1.0]
    @testset "GSL_Hybrids" begin
        ca = GSL_MultirootFSolverCache(GSL_Hybrids, f!, length(x0))
        r0 = solve!(ca, x0, verbose=true)
        r1 = solve!(ca, f!, x0, verbose=true)
        r2 = solve!(GSL_Hybrids, f!, x0)
        @test r0[1] == r1[1] == r2[1]
        @test r0[1] ≈ roots atol=1e-7
        r = solve!(ca, x0, verbose=5, ftol=0.01)
        @test r[1][2] ≈ 2.033707865566775 atol = 1e-6
    end
end

@testset "NLsolve" begin
    ss = rbcss()
    sol = [0.9900990099009883, 0.9658914728682162, 0.8816460975214576, 3.1428571428570864]
    f!(y, x) = residuals!(y, ss, x)
    r = solve!(NLsolve_Solver, f!, ss.inits)
    @test r[1] ≈ sol atol=1e-6
    @test r[2]
    r1 = solve!(NLsolve_Solver(), f!, ss.inits; method=:newton)
    @test r1[1] ≈ sol atol=1e-6

    ca = rootsolvercache(NLsolve_Solver, ss)
    r = solve!(ca, ss.inits)
    @test r[1] ≈ sol atol=1e-6
    @test r[2]
    ca = rootsolvercache(NLsolve_newton, ss)
    r = solve!(ca, ss.inits)
    @test r[1] ≈ sol atol=1e-6
    @test r[2]
    ca = rootsolvercache(NLsolve_trust_region, ss)
    r = solve!(ca, ss.inits)
    @test r[1] ≈ sol atol=1e-6
    @test r[2]
    ca = rootsolvercache(NLsolve_broyden, ss)
    r = solve!(ca, ss.inits)
    @test r[1] ≈ sol atol=1e-6
    @test r[2]
end

@testset "Roots" begin
    r = solve!(Roots_Default, sin, 3)
    @test r[1] ≈ 3.141592653589793 atol=1e-8
    @test r[2]
    r = solve!(Roots_Default(), sin, (3,4))
    @test r[1] ≈ 3.141592653589793 atol=1e-8
    @test r[2]
    r = solve!(Brent, sin, (3,4))
    @test r[1] ≈ 3.141592653589793 atol=1e-8
    @test r[2]
    r = solve!(Secant(), sin, 3)
    @test r[1] ≈ 3.141592653589793 atol=1e-8
    @test r[2]
end
