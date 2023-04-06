using SequenceJacobians.RBC
function rbcss()
    m = model(rbcblocks())
    calis = [:L=>1, :eis=>1, :frisch=>1, :δ=>0.025, :α=>0.11]
    tars = [:goods_mkt=>0, :r=>0.01, :euler=>0, :Y=>1]
    inits = [:φ=>0.9, :β=>0.99, :K=>2, :Z=>1]
    return SteadyState(m, calis, inits, tars)
end

@testset "NLsolve" begin
    ss = rbcss()
    sol = [0.9900990099009883, 0.9658914728682162, 0.8816460975214576, 3.1428571428570864]
    f!(y, x) = residuals!(y, ss, x)
    r = solve(NLsolve_Solver, f!, ss.inits)
    @test r[1] ≈ sol atol=1e-6
    @test r[2]
    r1 = solve(NLsolve_Solver(), f!, ss.inits; method=:newton)
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
