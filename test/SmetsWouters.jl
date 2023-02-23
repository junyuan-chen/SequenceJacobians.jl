@testset "SmetsWouters" begin
    using SequenceJacobians: SmetsWouters as sw
    data = exampledata(:sw)[75:end,:]
    # Manually specified intercepts are taken for the results to be comparable
    for n in (:y, :I, :c, :w)
        data[:,n] .-= 0.509
    end
    data[:,:pi] .-= 0.635
    data[:,:i] .-= 1.427
    data[:,:n] .-= 1.743

    vals = sw.swparams(sw.default_params)
    # Compare results with Python paper replication
    @test vals[:βbar] ≈ 0.997234733911248 atol=1e-10
    @test vals[:Rkss] ≈ 0.032877048223083016 atol=1e-10
    @test vals[:wss] ≈ 0.6118839873887233 atol=1e-10
    @test vals[:whlc] ≈ 0.836850535646599 atol=1e-10
    @test vals[:c3] ≈ 0.08414483893933412 atol=1e-10
    @test vals[:I2] ≈ 0.07560004268664199 atol=1e-10
    @test vals[:k2] ≈ 0.19827414450360034 atol=1e-10
    @test vals[:π3] ≈ 0.04405992150062866 atol=1e-10
    @test vals[:w4] ≈ 0.0060783524655172606 atol=1e-12
    @test vals[:ibar] ≈ 1.4277067479299577 atol=1e-10

    m, ss = sw.swmodelss(vals)
    endos = [:w, :wf, :n, :nf, :r, :rf]
    exos = [:εa, :εb, :εg, :εI, :εi, :εp, :εw]
    tars = [:goods_mkt_r, :goods_mkt_r_f, :fisher_r, :w_r, :μp_f, :μw_f]
    T = 300
    j = TotalJacobian(m, vcat(endos, exos), tars, ss[], T)

    # Compare results with Python paper replication
    Jc_εb = Matrix(j.parts[:c][:εb][1])
    @test Jc_εb[1:3,1] ≈ [1.79992836, 1.43981375, 1.15174786] atol=1e-7
    Jμw_c = j.parts[:μw][:c][1].S
    @test Jμw_c.v[1] ≈ -4.99820976 atol=1e-7
    JI_εI = Matrix(j.parts[:I][:εI][1])
    @test JI_εI[1,1:3] ≈ [1.78430936, 1.60521749, 1.43622494] atol=1e-7
    Jy_εa = Matrix(j.parts[:y][:εa][1])
    @test Jy_εa[1,1:3] ≈ [1.77, 0, 0]
    Jq_r = Matrix(j.parts[:q][:r][1])
    @test Jq_r[1,1:3] ≈ [-9.95427079e-1, -9.54409371e-1, -9.11310638e-1] atol=1e-8
    Jrkf_nf = Matrix(j.parts[:rkf][:nf][1])
    @test Jrkf_nf[2,1:3] ≈ [0, 4.00978458e-1, -4.02626029e-5] atol=1e-8
    Jμp_f_εb = Matrix(j.parts[:μp_f][:εb][1])
    @test Jμp_f_εb[2,1:3] ≈ [3.85569079e-3, 7.20627705e-3, 1.00890546e-2] atol=1e-9
    Jkf_wf = Matrix(j.parts[:kf][:wf][1])
    @test Jkf_wf[1,1:3] ≈ [0, 5.37215728e-5, 1.00405494e-4] atol=1e-11
    Jyf_wf = Matrix(j.parts[:yf][:wf][1])
    @test Jyf_wf[2,1:3] ≈ [0, 2.08872822e-1, 1.40391670e-5] atol=1e-8
    Ji_yf = Matrix(j.parts[:i][:yf][1])
    @test Ji_yf[1:3,1] ≈ [-0.14, 0.0025, 0.0021875]

    # Compare results with Python paper replication
    gj = GEJacobian(j, exos)
    Gw_εa = getM!(gj, :εa, :w)
    @test Gw_εa[1,1:3] ≈ [1.06775109e-2, 1.76690265e-2, 2.22590092e-2] atol=1e-9
    Gwf_εa = getM!(gj, :εa, :wf)
    @test Gwf_εa[1,1:3] ≈ [1.21353964, -3.71612075e-3, -2.81034302e-3] atol=1e-7
    @test Gwf_εa[T,T-2:T] ≈ [-1.15241184, -1.92334259, -1.98426882] atol=1e-7
    Gn_εa = getM!(gj, :εa, :n)
    @test Gn_εa[1,1:3] ≈ [-1.24746557, 3.61127208e-2, 3.70628445e-2] atol=1e-7
    Gnf_εa = getM!(gj, :εa, :nf)
    @test Gnf_εa[1,1:3] ≈ [-8.90392290e-1, 4.14902762e-2, 3.13773196e-2] atol=1e-8
    Gr_εa = getM!(gj, :εa, :r)
    @test Gr_εa[1,1:3] ≈ [-1.19277566e-1, 4.84551381e-2, 4.50415838e-2] atol=1e-8
    Grf_εa = getM!(gj, :εa, :rf)
    @test Grf_εa[1,1:3] ≈ [-6.66720126, 5.75552765, -1.38001597e-1] atol=1e-7

    Gw_εb = getM!(gj, :εb, :w)
    @test Gw_εb[1,1:3] ≈ [1.22197868e-1, 2.04922287e-1, 2.54091451e-1] atol=1e-8
    Gn_εb = getM!(gj, :εb, :n)
    @test Gn_εb[T,T-2:T] ≈ [-1.00531809, -1.29636016, -1.64413090] atol=1e-7
    Gwf_εg = getM!(gj, :εg, :wf)
    @test Gwf_εg[1,1:3] ≈ [-4.68091569e-2, 1.63828811e-3, 1.22766667e-3] atol=1e-9
    Gnf_εg = getM!(gj, :εg, :nf)
    @test Gnf_εg[T,T-2:T] ≈ [-3.48262567e-1, -6.00105658e-1, -3.63899054e-1] atol=1e-8
    Gw_εI = getM!(gj, :εI, :w)
    @test Gw_εI[1,1:3] ≈ [1.75589774e-2, 2.32056223e-2, 2.30568032e-2] atol=1e-9
    Gn_εI = getM!(gj, :εI, :n)
    @test Gn_εI[T,T-2:T] ≈ [-9.18030918e-2, -3.97120777e-2, 8.36281723e-2] atol=1e-9
    Gn_εi = getM!(gj, :εi, :n)
    @test Gn_εi[1,1:3] ≈ [-4.51560782e-1, -4.27857364e-1, -3.97114512e-1] atol=1e-8
    Gr_εi = getM!(gj, :εi, :r)
    @test Gr_εi[T,T-2:T] ≈ [7.86180210e-1, 9.06369101e-1, 1.02724363] atol=1e-7
    Gw_εp = getM!(gj, :εp, :w)
    @test Gw_εp[1,1:3] ≈ [-1.00502752, -8.58363818e-1, -7.32275843e-1] atol=1e-7
    Gr_εp = getM!(gj, :εp, :r)
    @test Gr_εp[T,T-2:T] ≈ [1.43968283,  1.71148689,  1.71918314] atol=1e-7
    Gw_εw = getM!(gj, :εw, :w)
    @test Gw_εw[1,1:3] ≈ [1.49434776, 1.08749850, 7.64572057e-1] atol=1e-7
    Gr_εw = getM!(gj, :εw, :r)
    @test Gr_εw[T,T-2:T] ≈ [1.02114307, 8.14215533e-1, 4.86547417e-1] atol=1e-7

    GI_εI = getM!(gj, :εI, :I)
    @test GI_εI[1,1:3] ≈ [1.71692803, 1.44189113, 1.18483426] atol=1e-7

    shocks = sw.swshocks()
    priors = sw.swpriors()
    obs = [:dy=>:y, :dI=>:I, :dc=>:c, :πp=>:pi, :i=>:i, :n=>:n, :dw=>:w]
    bm = bayesian(gj, shocks, obs, priors, data, demean=false)
    θ0 = [(0.4 for _ in 1:7)..., (0.8 for _ in 1:9)...]
    @time θmode, rx, niter, r = mode(bm, :LD_LBFGS, θ0, lower_bounds=0.01, upper_bounds=1,
        ftol_abs=1e-10, verbose=true)
    # Compare results with Python paper replication (default SciPy settings)
    pyr = [0.44591128, 0.97810607, 0.24588503, 0.25064353, 0.58909547,
        0.97055036, 0.46104506, 0.66256257, 0.22929228, 0.08587141,
        0.13492305, 0.97487284, 0.74004929, 0.25588583, 0.97553796, 0.92490134]
    iσ = union(1:2:11,14)
    @test rx[1:7] ≈ pyr[iσ] atol=1e-2
    @test rx[8:end] ≈ pyr[setdiff(1:16, iσ)] atol=1e-2

    tr = as(NamedTuple{keys(bm[])}(ntuple(i->as(Real,0.01,1), 16)))
    bm1 = transform(tr, bm)
    @time θmode1, rx1, niter1, r1 = mode(bm1, :LD_LBFGS, θ0,
        lower_bounds=-5, upper_bounds=5, ftol_abs=1e-10, verbose=30)
    @test collect(θmode1) ≈ rx atol=1e-4
end
