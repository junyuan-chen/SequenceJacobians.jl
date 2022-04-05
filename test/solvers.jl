@testset "GSL" begin
    n = 5
    roots = collect(1.0:5.0)
    function f!(out, x)
        out .= (x .- roots).^2
    end
    x0 = Float64[1.0, 5.0, 2.0, 1.5, -1.0]
    @testset "GSL_Hybrids" begin
        ca = GSL_MultirootFSolverCache(GSL_Hybrids, length(x0))
        r0 = solve!(ca, f!, x0, verbose=true)
        r1 = solve!(GSL_Hybrids, f!, x0)
        @test r0[1] == r1[1]
        @test r1[1] ≈ roots atol=1e-7
        r = solve!(ca, f!, x0, verbose=5, ftol=0.01)
        @test r[1][2] ≈ 2.033707865566775 atol = 1e-6
    end
end

@testset "Roots" begin
    r = solve!(Roots_Default_Solver, sin, 3)
    @test r[1] ≈ 3.141592653589793 atol=1e-8
    @test r[2]
    r = solve!(Roots_Solver{Brent}, sin, (3,4))
    @test r[1] ≈ 3.141592653589793 atol=1e-8
    @test r[2]
end
