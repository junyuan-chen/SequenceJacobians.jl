@testset "Shift" begin
    @test Lag() == Shift(-1)
    @test Lead() == Shift(1, 1.0)
    M = zeros(3, 3)
    M[3,2] = 1.0
    @test Shift((-1,1)=>1.0)(3) == M
    S = Lag(true)
    @test eltype(S) == Bool
    @test ndims(S) == 2
    @test has_offset_axes(S) == false
    @test copy(S) == S
    S1 = convert(Shift{Int}, S)
    @test S1 == Lag(1) && eltype(S1) == Int
    @test isdiag(S) == false
    @test isdiag(Shift(0))
    @test iszero(S) == false
    S2 = Shift((2,1)=>0.0)
    @test iszero(S2)
    @test transpose(S2) == Shift((-2,1)=>0)

    S = Shift((-1,1)=>2.0)
    @test +S == S
    @test S + S == Shift((-1,1)=>4.0)
    M = ones(3,3)
    M[3,2] += 2
    @test ones(3,3) + S == M

    @test -S == Shift((-1,1)=>-2.0)
    @test S - S == Shift((-1,1)=>0.0)
    M = -ones(3,3)
    M[3,2] += 2
    @test S - ones(3,3) == M
    M = ones(3,3)
    M[3,2] -= 2
    @test ones(3,3) - S == M

    @test 2 * S == Shift((-1,1)=>4.0)
    @test S / 2 == Shift((-1,1)=>1.0)

    @test S * S == Shift((-2,1)=>4.0)

    C = fill(NaN, 4, 4)
    B = I(4)
    mul!(C, S, B, 2.0, false)
    M = zeros(4, 4)
    M[[7,12]] .= 4
    @test C == M

    @test S * (1:4) == [0, 0, 4, 6]
end

@testset "ShiftMap" begin
    S = ShiftMap(Shift((1,0)=>1.0), 4)
    @test S == LinearMap(Shift((1,0)=>1.0), 4)
    U = LinearMap(2.0*I, 4)

    @test transpose(S) == ShiftMap(Shift((-1,0)=>1.0), 4)

    S1 = ShiftMap(Shift((-1,1)=>2.0), 4)
    @test S + S1 == ShiftMap(Shift((-1,1)=>2.0, (1,0)=>1.0), 4)
    @test S + U == U + S == ShiftMap(Shift((0,0)=>2.0, (1,0)=>1.0), 4)
    @test S - S1 == ShiftMap(Shift((-1,1)=>-2.0, (1,0)=>1.0), 4)
    @test S * 2 == ShiftMap(Shift((1,0)=>2.0), 4)

    @test S * U isa ShiftMap
    @test U * S isa ShiftMap
    @test S * S isa ShiftMap

    zmap = LinearMap(0.0*I, 4)
    @test S * zmap == zmap

    @test U + U isa UniformScalingMap
    @test U * U isa UniformScalingMap
    @test zero(S) == zmap

    A = LinearMap(rand(4,4))
    C = A * S
    B = C * S
    @test length(B.maps) == 2
    @test B.maps[1] == S * S
    @test length(((A * A) * S).maps) == 3

    C = S * A
    B = S * C
    @test length(B.maps) == 2
    @test B.maps[2] == S * S
    @test length((S * (A * A)).maps) == 3

    S = Shift((-1,1)=>2.0)
    M = convert(Matrix, LinearMap(S, 4))
    @test findall(view(M,:).==2) == [7, 12]
end
