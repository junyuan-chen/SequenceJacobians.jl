@testset "Horvath" begin
    using SequenceJacobians: Horvath as hv
    vlw = loadjson("vlw")
    p = hv.HorvathPlanner(vlw)
    N = length(p.C)
    T = 60
    mss = model([hv.ss_blk()])
    @testset "hwelast=-1" begin
        calis = Dict(:p=>p, :β=>0.96, :eis=>1, :frisch=>0, :hwelast=>-1)
        ss = SteadyState(mss, calis)
        @test log(ss[:Ctot]) ≈ -9.179427352597520 atol=1e-8
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
        vals = merge(ss[], (goods_mkt=zeros(N), euler=zeros(N), sA=1.0))
        @time J = TotalJacobian(m, (:A, :K, :μ), (:euler, :goods_mkt), vals, T)
        @time gj = GEJacobian(J, :A)
        gs = GMaps(gj, gj.endosrcs)
        irf = impulse(gs, :A=>0.01 .* p.ρA'.^(0:T-1), (:K, :μ))

        irfAK = irf[:A][:K][:,2,2] .+ vals[:K][2]
        @test irfAK[1:4] ≈ [8.37086769055079e-5, 8.37153354856315e-5, 8.37218320155476e-5,
            8.37233714224442e-5] atol=1e-9
        irfAK = irf[:A][:K][:,20,30] .+ vals[:K][20]
        @test irfAK[1:4] ≈ [1.21087421904459e-5, 1.21136555611731e-5, 1.21147035034710e-5,
            1.21145389348198e-5] atol=1e-10
        irfAμ = irf[:A][:μ][:,1,1] .+ vals[:μ][1]
        @test irfAμ[5:8] ≈ [1271.35797617984, 1272.05717332078, 1272.87106381285,
            1273.73236354833] atol=1e-1
        irfAμ = irf[:A][:μ][:,10,15] .+ vals[:μ][10]
        @test irfAμ[1:4] ≈ [926.915440844600, 926.911409549962, 926.906755224990,
            926.910364350904] atol=1e-3
    end

    if !Sys.iswindows() # Skip GitHub CI on Windows due to OutOfMemoryError
    @testset "hwelast=-1.04" begin
        calis = Dict(:p=>p, :β=>0.96, :eis=>1, :frisch=>0, :hwelast=>-1.04)
        ss = SteadyState(mss, calis)
        # Compare results from Dynare
        @test log(ss[:Ctot]) ≈ -9.121497724560054 atol=1e-8
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
        vals = merge(ss[], (goods_mkt=zeros(N), euler=zeros(N), sA=1.0))
        @time J = TotalJacobian(m, (:A, :K, :μ), (:euler, :goods_mkt), vals, T)
        @time gj = GEJacobian(J, :A)
        gs = GMaps(gj)
        A = 0.01 .* p.ρA'.^(0:T-1)
        @time irf = impulse(gs, :A=>A)

        irfΑμ = irf[:A][:μ][:,1,1] .+ vals[:μ][1]
        @test irfΑμ[5:8] ≈ [1193.32690087411, 1193.94423057912, 1194.64872211859,
            1195.39208371279] atol=1e-1
        irfΑμ = irf[:A][:μ][:,10,15] .+ vals[:μ][10]
        @test irfΑμ[1:4] ≈ [870.205098428633, 870.196228865397, 870.195230527473,
            870.197126094683] atol=1e-3
        irfAK = irf[:A][:K][:,2,2] .+ vals[:K][2]
        @test irfAK[1:4] ≈ [9.86631447086790e-5, 9.86676471509673e-5, 9.86707281854642e-5,
            9.86725041700443e-5] atol=1e-9
        irfAK = irf[:A][:K][:,20,30] .+ vals[:K][20]
        @test irfAK[1:4] ≈ [1.50685925523843e-5, 1.50698233156049e-5, 1.50706351640268e-5,
            1.50710416824670e-5] atol=1e-10

        irfL = sum(irf[:A][:L], dims=3) .+ vals[:L]'
        @test irfL[11:14,1] ≈ [0.0147524215916864, 0.0147483879991443, 0.0147469008443326,
            0.0147472869978414] atol=1e-6
        @test irfL[5:8,8] ≈ [0.0227951676237182, 0.0226526489315170, 0.0225581427283472,
            0.0224985611496701] atol=1e-6

        irfLtot = sum(irf[:A][:Ltot], dims=3) .+ vals[:Ltot]
        @test irfLtot[1:10] ≈ [0.804494965807056, 0.798088178218045, 0.793100997885200,
            0.789247588847981, 0.786310398275455, 0.784111645310415, 0.782503353260691,
            0.781362343396361, 0.780586698192282, 0.780092794081647] atol=1e-3
        @test sum(irf[:A][:L], dims=2) ≈ irf[:A][:Ltot] atol=1e-7

        irfCtot = sum(irf[:A][:Ctot], dims=3) .+ vals[:Ctot]
        @test irfCtot[1:4] ≈ [0.000110624106620004, 0.000110483994704759,
            0.000110458811679267, 0.000110457457876636] atol=1e-7

        irfY = sum(irf[:A][:Y], dims=3) .+ vals[:Y]'
        @test irfY[1:4,1] ≈ [5.35005037767409e-5, 5.32817262601225e-5,
            5.30932723698108e-5, 5.29287581415065e-5] atol=1e-7
        @test irfY[1:4,8] ≈ [9.56987436091410e-5, 9.36016594496518e-5,
            9.21166171061876e-5, 9.10338230540050e-5] atol=1e-6

        irf2 = impulse(gs, :A=>A, :VA, transform=true)
        irfVA = sum(irf2[:A][:VA], dims=3)
        @test irfVA[1:4,3] ≈ [11.8984806246017, 10.3439538558011, 8.30598923382437,
            6.29891065021744] atol=1
        @test irfVA[15:18,25] ≈ [-0.00352893117446795, -0.0165687244357060,
            -0.0250684500164411, -0.0302002285083591] atol=1e-3

        df = DataFrame(aslongtable(irf))
        @test nrow(df) == 825840
        @test df[(df.exovar.==:A37).&(df.endovar.==:K1),:value] == irf[:A][:K][:,1,37]

        εA = vcat(zeros(T-1,37), reshape(vlw[:εA],70,37))
        shocks = map(x->ARMAProcess(x, ()), p.ρA)
        s = simulate(gs, :A, :Ltot, εA, shocks)
        @test size(s) == (70, 1)
        @test s[1:4] ≈ [0.778555753205377, 0.805221510238579, 0.789908637509037,
            0.761696915244395] atol=1e-3

        s = simulate(gs, :A, :L, εA, shocks)
        @test size(s) == (70, 37)
        @test s[1:4,1] ≈ [0.0138751865364403, 0.0148416066788429,
            0.0147788643609457, 0.0140718615393009] atol=1e-4
        @test s[11:14,8] ≈ [0.0206388724551132, 0.0201142193464190,
            0.0211262835460427, 0.0243975519413298] atol=1e-3

        dA = vec(A)
        j1 = TotalJacobian(m, (:A, :K, :μ), (:euler, :goods_mkt), vals, T, dZs=(:A=>dA,))
        @test j1.ncol == [1, 2220, 2220]
        gj1 = GEJacobian(j1, :A)
        gs1 = GMaps(gj1)
        tarvals = rowblocks(zeros(N*16), 16)
        tarblk = rowblocks(gs1(:A, :Y), T)
        @test reshape(tarblk,T,N) ≈ sum(irf[:A][:Y], dims=3)
        for s in 1:N
            tarvals[Block(s)] .= view(view(tarblk, Block(s)), 1:16)
        end
        u1 = ImpulseUpdate(gs1, :sA, :A, :Y, 16)
        md = ImpulseResidual(u1, tarvals)
        @test u1[] == (sA=1.0,)
        fdf = OnceDifferentiable(md, zeros(1), zeros(length(tarvals)))
        r1 = solve(Hybrid{LeastSquares}, fdf, [1.1], showtrace=1)
        @test Symbol(getexitstate(r1)) == :ftol_reached
    end
    end
end
