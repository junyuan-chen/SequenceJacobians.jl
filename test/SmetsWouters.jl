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

    m, ss = sw.swmodelss(vals, Hybrid)
    endos = [:w, :wf, :n, :nf, :r, :rf]
    exos = [:εa, :εb, :εg, :εI, :εi, :εp, :εw]
    tars = [:goods_mkt_r, :goods_mkt_r_f, :fisher_r, :w_r, :μp_f, :μw_f]
    T = 300
    j = TotalJacobian(m, vcat(endos, exos), tars, ss[], T)

    # Compare results with Python paper replication
    Jc_εb = j.parts[:c][:εb]
    @test Jc_εb[1:3,1] ≈ [1.79992836, 1.43981375, 1.15174786] atol=1e-7
    Jμw_c = j.parts[:μw][:c](3)
    @test Jμw_c[:,1] ≈ [-4.99820976, 3.99820976, 0] atol=1e-7
    JI_εI = j.parts[:I][:εI]
    @test JI_εI[1,1:3] ≈ [1.78430936, 1.60521749, 1.43622494] atol=1e-7
    Jy_εa = j.parts[:y][:εa]
    @test Jy_εa[1,1:3] ≈ [1.77, 0, 0]
    Jq_r = j.parts[:q][:r]
    @test Jq_r[1,1:3] ≈ [-9.95427079e-1, -9.54409371e-1, -9.11310638e-1] atol=1e-8
    Jrkf_nf = j.parts[:rkf][:nf]
    @test Jrkf_nf[2,1:3] ≈ [0, 4.00978458e-1, -4.02626029e-5] atol=1e-8
    Jμp_f_εb = j.parts[:μp_f][:εb]
    @test Jμp_f_εb[2,1:3] ≈ [3.85569079e-3, 7.20627705e-3, 1.00890546e-2] atol=1e-9
    Jkf_wf = j.parts[:kf][:wf]
    @test Jkf_wf[1,1:3] ≈ [0, 5.37215728e-5, 1.00405494e-4] atol=1e-11
    Jyf_wf = j.parts[:yf][:wf]
    @test Jyf_wf[2,1:3] ≈ [0, 2.08872822e-1, 1.40391670e-5] atol=1e-8
    Ji_yf = j.parts[:i][:yf]
    @test Ji_yf[1:3,1] ≈ [-0.14, 0.0025, 0.0021875]

    # Compare results with Python paper replication
    gj = GEJacobian(j, exos)
    gs = GMaps(gj)
    G = zeros(T, T)
    Gw_εa = gs(G, :εa, :w)
    @test Gw_εa[1,1:3] ≈ [1.06775109e-2, 1.76690265e-2, 2.22590092e-2] atol=1e-9
    Gwf_εa = gs(G, :εa, :wf)
    @test Gwf_εa[1,1:3] ≈ [1.21353964, -3.71612075e-3, -2.81034302e-3] atol=1e-7
    @test Gwf_εa[T,T-2:T] ≈ [-1.15241184, -1.92334259, -1.98426882] atol=1e-7
    Gn_εa = gs(G, :εa, :n)
    @test Gn_εa[1,1:3] ≈ [-1.24746557, 3.61127208e-2, 3.70628445e-2] atol=1e-7
    Gnf_εa = gs(G, :εa, :nf)
    @test Gnf_εa[1,1:3] ≈ [-8.90392290e-1, 4.14902762e-2, 3.13773196e-2] atol=1e-8
    Gr_εa = gs(G, :εa, :r)
    @test Gr_εa[1,1:3] ≈ [-1.19277566e-1, 4.84551381e-2, 4.50415838e-2] atol=1e-8
    Grf_εa = gs(G, :εa, :rf)
    @test Grf_εa[1,1:3] ≈ [-6.66720126, 5.75552765, -1.38001597e-1] atol=1e-7

    Gw_εb = gs(G, :εb, :w)
    @test Gw_εb[1,1:3] ≈ [1.22197868e-1, 2.04922287e-1, 2.54091451e-1] atol=1e-8
    Gn_εb = gs(G, :εb, :n)
    @test Gn_εb[T,T-2:T] ≈ [-1.00531809, -1.29636016, -1.64413090] atol=1e-7
    Gwf_εg = gs(G, :εg, :wf)
    @test Gwf_εg[1,1:3] ≈ [-4.68091569e-2, 1.63828811e-3, 1.22766667e-3] atol=1e-9
    Gnf_εg = gs(G, :εg, :nf)
    @test Gnf_εg[T,T-2:T] ≈ [-3.48262567e-1, -6.00105658e-1, -3.63899054e-1] atol=1e-8
    Gw_εI = gs(G, :εI, :w)
    @test Gw_εI[1,1:3] ≈ [1.75589774e-2, 2.32056223e-2, 2.30568032e-2] atol=1e-9
    Gn_εI = gs(G, :εI, :n)
    @test Gn_εI[T,T-2:T] ≈ [-9.18030918e-2, -3.97120777e-2, 8.36281723e-2] atol=1e-9
    Gn_εi = gs(G, :εi, :n)
    @test Gn_εi[1,1:3] ≈ [-4.51560782e-1, -4.27857364e-1, -3.97114512e-1] atol=1e-8
    Gr_εi = gs(G, :εi, :r)
    @test Gr_εi[T,T-2:T] ≈ [7.86180210e-1, 9.06369101e-1, 1.02724363] atol=1e-7
    Gw_εp = gs(G, :εp, :w)
    @test Gw_εp[1,1:3] ≈ [-1.00502752, -8.58363818e-1, -7.32275843e-1] atol=1e-7
    Gr_εp = gs(G, :εp, :r)
    @test Gr_εp[T,T-2:T] ≈ [1.43968283,  1.71148689,  1.71918314] atol=1e-7
    Gw_εw = gs(G, :εw, :w)
    @test Gw_εw[1,1:3] ≈ [1.49434776, 1.08749850, 7.64572057e-1] atol=1e-7
    Gr_εw = gs(G, :εw, :r)
    @test Gr_εw[T,T-2:T] ≈ [1.02114307, 8.14215533e-1, 4.86547417e-1] atol=1e-7

    GI_εI = gs(G, :εI, :I)
    @test GI_εI[1,1:3] ≈ [1.71692803, 1.44189113, 1.18483426] atol=1e-7

    shocks = sw.swshocks()
    priors = sw.swpriors()
    obs = [:dy=>:y, :dI=>:I, :dc=>:c, :πp=>:pi, :i=>:i, :n=>:n, :dw=>:w]
    bm = bayesian(gs, shocks, obs, priors, data, demean=false)
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

    p = plan(gj, :ρ)
    @test sprint(show, p) == "GEJacobianUpdatePlan(1)"
    @test sprint(show, MIME("text/plain"), p) == """
        GEJacobianUpdatePlan with 1 block jacobian:
          CombinedBlockJacobian(taylor_r: πp, y, yf, εi → i)"""
    p = plan(gj, :α)
    @test sprint(show, p) == "GEJacobianUpdatePlan(2)"
    @test sprint(show, MIME("text/plain"), p) == """
        GEJacobianUpdatePlan with 2 block jacobians:
          CombinedBlockJacobian(k_r_f, rk_r_f, q_r_f, I_r_f: εI, rf, εb, nf, wf, εa → If, qf, rkf, kf, ksf, zf, yf, μp_f)
          CombinedBlockJacobian(k_r, rk_r, q_r, I_r: εI, r, εb, n, w, εa → I, q, rk, k, ks, z, y, μp)"""
    @test sprint(show, gs) == "GMaps{Float64}(εa, εb, εg, εI, εi, εp, εw)"

    εi = zeros(T, 1)
    εi[1] = 1
    j1 = TotalJacobian(m, vcat(endos, exos), tars, ss[], T, dZs=[:εi=>εi])
    @test j1.totals[:εi][:i].out ≈ j1.parts[:i][:εi][:,1]
    gj1 = GEJacobian(j1, :εi, endos)
    gs1 = GMaps(gj1)
    vobs = [:y, :I, :c, :i, :n, :w]
    vars = [:εi=>vobs]
    nobs = length(vobs)
    # Use actual irfs under original parameters as targets
    tarvals = PseudoBlockVector(Vector{Float64}(undef, 16*nobs), fill(16, nobs))
    for (i, n) in enumerate(vobs)
        gs1(view(tarvals, Block(i)), :εi, n)
    end
    u = ImpulseUpdate(gs1, :ρ, :εi, vobs, 16)
    function md(resids, θ)
        u(θ)
        resids .= _reshape(u.vals, length(u.vals)) .- tarvals.blocks
    end
    @test u[] == (ρ=0.875,)
    fdf = OnceDifferentiable(md, [0.6], zeros(length(tarvals)))
    r = solve(Hybrid{LeastSquares}, fdf, [0.6])
    @test u[:ρ] == r.x[1]
    @test u[:ρ] ≈ 0.875 atol=1e-7

    εa = zeros(T, 1)
    εi[1] = 1
    j2 = TotalJacobian(m, vcat(endos, [:εa, :εi]), tars, ss[], T, dZs=[:εa=>εa, :εi=>εi])
    gj2 = GEJacobian(j2, (:εa, :εi), endos)
    gs2 = GMaps(gj2)
    vars = [:εa=>vobs, :εi=>vobs]
    tarvals = PseudoBlockVector(Vector{Float64}(undef, 32*nobs), fill(16, 2*nobs))
    i = 1
    for (exo, vobs) in vars
        for n in vobs
            gs2(view(tarvals, Block(i)), exo, n)
            i += 1
        end
    end
    u2 = ImpulseUpdate(gs2, (:α, :ρ), (:εa, :εi), vobs, 16)
    md2 = ImpulseResidual(u2, tarvals.blocks)
    @test u2[] == (α=0.197, ρ=0.875)
    fdf = OnceDifferentiable(md2, zeros(2), zeros(length(tarvals)))
    r1 = solve(Hybrid{LeastSquares}, fdf, [0.4, 0.6], thres_jac=1)
    @test u2[:α] ≈ 0.197 atol=1e-6
    @test u2[:ρ] ≈ 0.875 atol=1e-6

    @test sprint(show, u) == "96×1 ImpulseUpdate{Float64}(1)"
    @test sprint(show, MIME("text/plain"), u) == """
        96×1 ImpulseUpdate{Float64} with 1 exogenous variable:
          parameter: ρ"""
    @test sprint(show, u2) == "192×2 ImpulseUpdate{Float64}(2)"
    @test sprint(show, MIME("text/plain"), u2) == """
        192×2 ImpulseUpdate{Float64} with 2 exogenous variables:
          parameters: α, ρ"""
end
