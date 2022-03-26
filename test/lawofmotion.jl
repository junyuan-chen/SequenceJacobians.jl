@testset "rouwenhorstexp" begin
    eproc = rouwenhorstexp(0.966, 0.5, 7)
    # Compare results with original Python package
    @test grid(eproc) ≈ [0.25952913, 0.39037867, 0.58720002, 0.88325488, 1.32857484, 1.99841649, 3.00597929] atol=1e-7
    @test eproc.d ≈ [0.015625, 0.09375, 0.234375, 0.3125, 0.234375, 0.09375, 0.015625] atol=1e-7
    @test eproc.m[1,:] ≈ [9.02237984e-01, 9.36198112e-02, 4.04765206e-03, 9.33334487e-05, 1.21058135e-06, 8.37431659e-09, 2.41375690e-11] atol=1e-7
    @test eproc.m[7,:] ≈ [2.41375690e-11, 8.37431659e-09, 1.21058135e-06, 9.33334487e-05, 4.04765206e-03, 9.36198112e-02, 9.02237984e-01] atol=1e-7
end

@testset "assetproc" begin
    aproc = assetproc(0, 200, 500, 500, 7)
    g = grid(aproc)
    @test g[1] == 0
    # Compare results with original Python package
    @test g[2] ≈ 3.37217033e-03 atol=1e-7
    @test g[101] ≈ 0.7046194802346136 atol=1e-7
    @test size(aproc.li) == (500, 7)
    @test size(aproc.lp) == (500, 7)
end
