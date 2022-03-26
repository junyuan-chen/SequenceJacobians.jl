@testset "supconverged" begin
    A = rand(4,4)
    B = copy(A)
    B[end,end] += 0.1
    @test !supconverged(A, B)
    @test supconverged(A, B, 1)
    @test_throws ArgumentError supconverged(A, rand(3,4))
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
