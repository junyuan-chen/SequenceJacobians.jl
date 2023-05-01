@testset "Shift" begin
    vv = ones(4, 4)
    zz = zeros(4, 4)
    S = Shift([(-1,1)], [[view(vv, 2:2, 1:1)]], (1,1))
    S1 = Shift([(-1,1)], [[view(vv, 3:3, 1:1)]], (1,1))
    S2 = Shift([(1,0)], [[view(vv, 3:3, 1:1)]], (1,1))
    Z = Shift([(-1,0),(2,1)], [[view(zz, 2:2, 1:1)], [view(zz, 1:1, 1:1)]], (1,1))
    @test iszero(Z)
    M = zeros(3, 3)
    M[3,2] = 1.0
    @test S(3) == M
    @test (Z + S)(3) == M
    @test eltype(S) == Float64
    @test ndims(S) == 2
    @test has_offset_axes(S) == false
    @test isdiag(S) == false
    @test isdiag(Shift([(0,1)], [[view(vv, 2:2, 1:1)]], (1,1)))
    @test iszero(S) == false

    M = S(5)
    @test (+S)(5) == M
    @test (S + S)(5) == 2*M
    S1 = S * true
    @test (S * S1)(5) == CompositeShift([(-2,1)], [1], (1,1))(5)
    @test sparse(S, 5) == M
    @test sparse(Z, 5) == Z(5)

    C = fill(NaN, 4, 4)
    B = I(4)
    mul!(C, S, B, 4.0, false)
    M = zeros(4, 4)
    M[[7,12]] .= 4
    @test C == M
    fill!(vv, 2.0)
    @test S * (1:4) == [0, 0, 4, 6]
end
