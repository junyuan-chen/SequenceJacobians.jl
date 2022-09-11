const vlw = loadjson("vlw")

function getsecirf(p, vals, GJ, z, u, i, j, T)
    A = 0.01 .* p.ρA[i].^(0:T-1)
    irf = fill(vals[u][j], T)
    return mul!(irf, GJ.Gs[z][u][j,i], A, true, true)
end

function getfullirf!(irf, p, GJ, z, u, T)
    N = length(p.ρA)
    A = reshape((0.01 .* p.ρA'.^(0:T-1)), N*T)
    return mul!(irf, GJ.Gs[z][u], A, true, true)
end

@testset "Horvath" begin
    using SequenceJacobians: Horvath as hv
    p = hv.HorvathPlanner(vlw)
    N = length(p.C)
    T = 100
    mss = model([hv.ss_blk()])
    @testset "hwelast=-1" begin
        calis = [:p=>p, :β=>0.96, :eis=>1, :frisch=>0, :hwelast=>-1]
        ss = SteadyState(mss, calis)
        @test log(getval(ss, :Ctot)) ≈ -9.179427352597520 atol=1e-8
        @test log(sum(p.L)) ≈ -0.245118891298135 atol=1e-8
        @test log(p.C[2]) ≈ -10.629366617908783 atol=1e-8
        @test log(p.L[2]) ≈ -4.471962580873086 atol=1e-8
        @test log(p.Y[2]) ≈ -9.789612124011620 atol=1e-8
        @test log(p.X[2]) ≈ -12.841456611394770 atol=1e-8
        @test log(p.K[2]) ≈ -9.388224783003304 atol=1e-8
        @test log(p.μ[2]) ≈ 6.932100778987180 atol=1e-8
        @test log(p.I[2]) ≈ -12.791799989451425 atol=1e-8
        @test log(p.ζ[2]) ≈ 8.209376375200970 atol=1e-8
        @test log(p.VA[2]) ≈ -7.758900353404101 atol=1e-8
        @test log(p.Z[3]) ≈ -8.703794453549149 atol=1e-8

        m = hv.Horvathmodel(p, calis)
        vals = getvarvals(ss)
        vals = merge(vals, (goods_mkt=zeros(N), euler=zeros(N), Ltotdiff=0.0))
        J = TotalJacobian(m, (:Y,:μ,:A), (:euler,:goods_mkt), vals, T)
        @time GJ = GEJacobian(J, :A)

        irf = getsecirf(p, vals, GJ, :A, :μ, 1, 1, T)
        @test irf[5:8] ≈ [1271.35797617984, 1272.05717332078, 1272.87106381285, 1273.73236354833] atol=5
        irf = getsecirf(p, vals, GJ, :A, :μ, 15, 10, T)
        @test irf[1:4] ≈ [926.915440844600, 926.911409549962, 926.906755224990,
            926.910364350904] atol=1e-1
        irf = getsecirf(p, vals, GJ, :A, :Y, 2, 2, T)
        @test irf[1:4] ≈ [5.66186331302610e-5, 5.64251076287809e-5, 5.62957924778920e-5,
            5.62099421245202e-5] atol=1e-7
        irf = getsecirf(p, vals, GJ, :A, :Y, 30, 20, T)
        @test irf[1:4] ≈ [1.62661395111125e-5, 1.62663065877140e-5, 1.62672989571761e-5,
            1.62664310552325e-5] atol=1e-7
    end

    @testset "hwelast=-1.04" begin
        calis = [:p=>p, :β=>0.96, :eis=>1, :frisch=>0, :hwelast=>-1.04]
        ss = SteadyState(mss, calis)
        # Compare results from Dynare
        @test log(getval(ss, :Ctot)) ≈ -9.121497724560054 atol=1e-8
        @test log(sum(p.L)) ≈ -0.245118891298135 atol=1e-8
        @test log(p.C[1]) ≈ -12.995980539507782 atol=1e-8
        @test log(p.L[1]) ≈ -4.212831002607052 atol=1e-8
        @test log(p.Y[1]) ≈ -9.868992366820157 atol=1e-8
        @test log(p.X[1]) ≈ -12.716894361909970 atol=1e-8
        @test log(p.K[1]) ≈ -9.978711995691427 atol=1e-8
        @test log(p.μ[1]) ≈ 7.093862732760659 atol=1e-8
        @test log(p.I[1]) ≈ -12.621928990695244 atol=1e-8
        @test log(p.ζ[1]) ≈ 8.190199066418831 atol=1e-8
        @test log(p.VA[1]) ≈ -7.443153329291874 atol=1e-8
        @test log(p.Z[1]) ≈ -11.992266967437140 atol=1e-8

        m = hv.Horvathmodel(p, calis)
        vals = getvarvals(ss)
        vals = merge(vals, (goods_mkt=zeros(N), euler=zeros(N), Ltotdiff=0.0))

        J = TotalJacobian(m, (:Y,:μ,:A), (:euler,:goods_mkt), vals, T)
        @time GJ = GEJacobian(J, :A)

        @test J.totals[:A][:ζ][1] isa ShiftMap
        @test J.totals[:Y][:Ltot][1] isa UniformScalingMap

        irf = getsecirf(p, vals, GJ, :A, :μ, 1, 1, T)
        @test irf[5:8] ≈ [1193.32690087411, 1193.94423057912, 1194.64872211859, 1195.39208371279] atol=4
        irf = getsecirf(p, vals, GJ, :A, :μ, 15, 10, T)
        @test irf[1:4] ≈ [870.217282656404, 870.205098428633, 870.196228865397,
            870.195230527473] atol=1e-1
        irf = getsecirf(p, vals, GJ, :A, :Y, 2, 2, T)
        @test irf[1:4] ≈ [6.19917974480857e-5, 6.17784670428334e-5, 6.16366958120238e-5,
            6.15422891727072e-5] atol=1e-7
        irf = getsecirf(p, vals, GJ, :A, :Y, 30, 20, T)
        @test irf[1:4] ≈ [1.74762447698397e-5, 1.74758764217527e-5, 1.74753134709252e-5,
            1.74746649195056e-5] atol=1e-7

        G = getG!(GJ, :A, :Ctot)
        irf = fill(vals[:Ctot], T)
        getfullirf!(irf, p, GJ, :A, :Ctot, T)
        # Responses are slightly stronger than results from Dynare because of μ
        @test irf[1:4] ≈ [0.000109314568592154, 0.000109312447761325, 0.000109311198505424,
            0.000109310207322562] atol=1e-6

        G = getG!(GJ, :A, :Ltot)
        irf = fill(vals[:Ltot], T)
        getfullirf!(irf, p, GJ, :A, :Ltot, T)
        @test irf[5:8] ≈ [0.782656255115046, 0.782619725392643, 0.782593500317328,
            0.782575882374397] atol=1e-2

        εA = vcat(zeros(99,37), reshape(vlw[:εA],70,37))
        s = simulate(GJ, :A, :Ltot, εA, p.ρA)
        @test size(s) == (70, 1)
        @test s[1:4] ≈ [0.778555753205377, 0.805221510238579, 0.789908637509037,
            0.761696915244395] atol=5e-2
        @test haskey(GJ.Ms[:A], :Ltot)

        s = simulate(GJ, :A, :L, εA, p.ρA)
        @test size(s) == (70, 37)
        @test s[1,1:3] ≈ [0.0138751865364403, 0.0113833972812861,
            0.0444779044081862] atol=5e-3
    end
end
