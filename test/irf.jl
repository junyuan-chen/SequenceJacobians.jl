@testset "linirf" begin
    ss = rbcss()
    f!(y,x) = residuals!(y, ss, x)
    solve!(GSL_Hybrids, ss, xtol=1e-10)
    J = TotalJacobian(model(ss), [:Z,:K,:L], [:euler, :goods_mkt], getvarvals(ss), 300)
    GJ = GEJacobian(J, :Z)

    dZ = zeros(300)
    dZ[11:end] .= getvarvals(ss)[:Z] .* 0.01 .* 0.8.^(0:289)
    irfs = linirf(GJ, :Z=>dZ)
    dC = irfs[:Z][:C]
    # Compare results with original Python package
    @test dC[1] ≈ 7.71383227e-4 atol=1e-11
    @test dC[end] ≈ 4.78652388e-18  atol=1e-20

    irfs2, GJ2 = linirf(J, :Z=>dZ, transform=[:K,:L,:Y,:C,:I,:w])
    dC2 = irfs2[:Z][:C]
    @test dC2[1] ≈ 8.37160091e-2 atol=1e-9
    @test dC2[end] ≈ 5.19467708e-16  atol=1e-18

    irfs3 = linirf(GJ, :Z=>dZ, transform=true)
    @test irfs3[:Z][:C] ≈ dC2

    tb = astable(irfs)
    @test length(tb[:Z]) == 8
    @test tb[:Z][:C] == dC
end
