@testset "mapmatmul" begin
    m = rand(3, 3)
    M = LinearMap(m)
    U = LinearMap(2.0 * I, 3)
    A = Matrix{WrappedMap}(undef, 2, 2)
    B = Matrix{LinearMap{Float64}}(undef, 2, 2)
    fill!(A, M)
    fill!(B, U)
    MMr = mapmatmul(A, B)
    @test size(MMr) == (2, 2)
    @test MMr.A === A
    @test MMr.rmap === B
    @test MMr.lmap === nothing
    @test MMr.amap === nothing

    C = Vector{Float64}(undef, 6)
    X = rand(6)
    mul!(C, MMr, X)
    @test C ≈ [m m; m m] * [U U; U U] * X

    U1 = LinearMap(2.0 * I, 3)
    A1 = Matrix{WrappedMap}(undef, 2, 1)
    fill!(A1, M)
    MMr1 = mapmatmul(A1, U1)
    @test size(MMr1) == (2, 1)
    @test MMr1.A === A1
    @test MMr1.rmap[1] === U1
    @test MMr1.lmap === nothing
    @test MMr1.amap === nothing

    X1 = rand(3)
    mul!(C, MMr1, X1)
    @test C ≈ [m; m] * U1 * X1

    MMl = mapmatmul(B, A)
    @test size(MMl) == (2, 2)
    @test MMl.A === A
    @test MMl.lmap === B
    @test MMl.rmap === nothing
    @test MMl.amap === nothing

    mul!(reshape(C,6,1), MMl, reshape(X,6,1))
    @test C ≈ [U U; U U] * [m m; m m] * X

    B1 = Matrix{LinearMap}(undef, 1, 2)
    fill!(B1, U)
    MMlr1 = mapmatmul(B1, MMr)
    @test size(MMlr1) == (1, 2)
    @test MMlr1.A === MMr.A
    @test MMlr1.lmap === B1
    @test MMlr1.rmap === MMr.rmap
    @test MMlr1.amap === nothing

    C1 = Vector{Float64}(undef, 3)
    mul!(C1, MMlr1, X)
    @test C1 ≈ [U U] * [m m; m m] * [U U; U U] * X

    B1 = reshape(B1, 2, 1)
    MMlr2 = mapmatmul(MMl, B1)
    @test size(MMlr2) == (2, 1)
    @test MMlr2.A === MMl.A
    @test MMlr2.lmap === MMl.lmap
    @test MMlr2.rmap === B1
    @test MMlr2.amap === nothing

    mul!(C, MMlr2, X1)
    @test C ≈ [U U; U U] * [m m; m m] * [U; U] * X1

    MM = mapmatmul(MMlr1, MMlr2)
    @test size(MM) == (1, 1)
    @test MM.A === MMlr1.A
    @test MM.lmap === MMlr1.lmap
    @test MM.rmap isa MatMulMap
    @test MM.amap === nothing

    mul!(C1, MM, X1)
    @test C1 ≈ [U U] * [m m; m m] * [U U; U U] * [U U; U U] * [m m; m m] * [U; U] * X1

    @test mapmatmul(nothing, nothing) === nothing

    MMra1 = MMr + B
    @test MMra1.amap === B
    mul!(C, MMra1, X)
    @test C ≈ [m m; m m] * [U U; U U] * X + [U U; U U] * X

    MMra2 = MMra1 + B
    @test MMra2.amap == B .+ B
    mul!(C, MMra2, X)
    @test C ≈ [m m; m m] * [U U; U U] * X + 2 * [U U; U U] * X

    MMra3 = MMra1 + MMra1
    mul!(C, MMra3, X)
    @test C ≈ 2 * [m m; m m] * [U U; U U] * X + 2 * [U U; U U] * X

    @test mapmatmul(B, B) == B * B
    @test mapmatmul(B1, U) == B1 .* Ref(U)

    Mat = Matrix(MMra1)
    C2 = similar(C)
    @test mul!(C2, Mat, X) ≈ mul!(C, MMra1, X)

    Mat = Matrix(MMra3)
    @test mul!(C2, Mat, X) ≈ mul!(C, MMra3, X)
end
