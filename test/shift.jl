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

