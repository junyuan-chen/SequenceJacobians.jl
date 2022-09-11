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
end

@testset "ShiftMap" begin
    S = ShiftMap(Shift((1,0)=>1.0), 4)
    U = LinearMap(2.0*I, 4)

    @test S + U isa ShiftMap
    @test U + S isa ShiftMap
    @test S + S isa ShiftMap

    @test S * U isa ShiftMap
    @test U * S isa ShiftMap
    @test S * S isa ShiftMap

    @test U + U isa UniformScalingMap
    @test U * U isa UniformScalingMap
    @test zero(S) == LinearMap(0.0*I, 4)
end
