@testset "impulse" begin
    ss = rbcss()
    solve(Hybrid, ss, ss.inits, ftol=1e-10)
    J = TotalJacobian(model(ss), [:Z,:K,:L], [:euler, :goods_mkt], ss[], 300)
    gj = GEJacobian(J, :Z)
    gs = GMaps(gj)

    dZ = zeros(300)
    dZ[11:end] .= ss[:Z] .* 0.01 .* 0.8.^(0:289)
    irfs = impulse(gs, :Z=>dZ)
    dC = irfs[:Z][:C]
    # Compare results with original Python package
    @test dC[1] ≈ 7.71383227e-4 atol=1e-10
    @test dC[end] ≈ 4.78652388e-18  atol=1e-20

    irfs2 = impulse(gs, :Z=>dZ, transform=[:K,:L,:Y,:C,:I,:w])
    dC2 = irfs2[:Z][:C]
    @test dC2[1] ≈ 8.37160091e-2 atol=1e-8
    @test dC2[end] ≈ 5.19467708e-16 atol=1e-18

    df = DataFrame(aswidetable(irfs))
    @test ncol(df) == 8
    @test df.Z_C == vec(dC)
end
