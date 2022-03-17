@testset "linfconverged" begin
    A = rand(4,4)
    B = copy(A)
    B[end,end] += 0.1
    @test !linfconverged(A, B)
    @test linfconverged(A, B, 1)
    @test_throws ArgumentError linfconverged(A, rand(3,4))
end

@testset "interpolate" begin
    x = 0:2
    y = exp.(x)
    xq = [-1, 1.1, 0.5, 2.5]
    yq = similar(xq)
    interpolate_y!(yq, xq, y, x)
    # Compare results with original Python package
    @test yq ≈ [-0.71828183, 3.18535926, 1.85914091, 9.72444323] atol=1e-7

    xq = [1, 5, 2.5, last(y)]
    xqi = zeros(Int, length(xq))
    xqpi = similar(xq)
    interpolate_coord!(xqi, xqpi, xq, y)
    # Compare results with original Python package
    # The Python code does not handle cases out of bounds
    @test xqi == [1,2,1,2]
    @test xqpi ≈ [1.0, 0.51149038, 0.12703494, 0.0] atol=1e-7
end

@testset "ExampleUtils" begin
    using SequenceJacobians.ExampleUtils

    @testset "grida" begin
        agrid = grida(200, 500)
        @test agrid[1] == 0
        # Compare results with original Python package
        @test agrid[2] ≈ 3.37217033e-03 atol=1e-7
        @test agrid[101] ≈ 0.7046194802346136 atol=1e-7
    end

    @testset "gridrouwenhorst" begin
        y, pr, Pi = gridrouwenhorst(0.966, 0.5, 7)
        # Compare results with original Python package
        @test y ≈ [0.25952913, 0.39037867, 0.58720002, 0.88325488, 1.32857484, 1.99841649, 3.00597929] atol=1e-7
        @test pr ≈ [0.015625, 0.09375, 0.234375, 0.3125, 0.234375, 0.09375, 0.015625] atol=1e-7
        @test Pi[1,:] ≈ [9.02237984e-01, 9.36198112e-02, 4.04765206e-03, 9.33334487e-05, 1.21058135e-06, 8.37431659e-09, 2.41375690e-11] atol=1e-7
        @test Pi[7,:] ≈ [2.41375690e-11, 8.37431659e-09, 1.21058135e-06, 9.33334487e-05, 4.04765206e-03, 9.36198112e-02, 9.02237984e-01] atol=1e-7
    end
end
