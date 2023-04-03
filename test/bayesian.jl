@testset "bayesian" begin
    data = exampledata(:bayes)
    data[:,:y] ./= 4
    using SequenceJacobians.KrusellSmith
    m = model(ksblocks())
    calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
    tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
    inits = [:β=>0.98, :Z=>0.85, :K=>3]
    ss = SteadyState(m, calis, inits, tars)
    solve(Hybrid, ss, ss.inits, ftol=1e-10)
    j = TotalJacobian(m, [:Z,:K], [:asset_mkt], ss[], 300, excluded=(:goods_mkt,))
    gs = GMaps(GEJacobian(j, :Z))
    shock, priors = kspriors()

    bm = bayesian(gs, shock, :Y=>:y, priors, data)
    @test dimension(bm) == 3
    θ = [0.2, 0.9, 0.003]
    lpri = logprior(bm, θ)
    lpost = logposterior!(bm, θ)
    @test bm[] == (σ = 0.2, ar = 0.9, ma = 0.003)
    @test bm[:σ] == 0.2
    # Compare results with Python paper replication
    # log prior is not comparable as the Python code does not add constant terms
    @test lpost - lpri - nrow(data)*log(2*pi)/2 ≈ 25.315994892433707 atol=1e-4
    @test logposterior!(bm, (θ...,)) ≈ lpost
    l, dl = logposterior_and_gradient!(bm, θ)
    @test l ≈ lpost
    @test dl ≈ [-166.70030975341797, 1.4917106628417969, 504.52807235717773] atol=1e-4
    l1, dl1 = logdensity_and_gradient(bm, θ)
    @test l1 ≈ l
    @test dl1 ≈ dl
    @test dl1 !== dl
    bm1 = bayesian(gs, shock, :Y=>:y, priors, data, fdtype=Val(:central))
    l1, dl1 = logposterior_and_gradient!(bm1, θ)
    @test dl1 ≈ dl atol=1e-2

    θ0 = [0.4, 0.5, 0.4]
    θmode, rx, niter, r = mode(bm, :LD_LBFGS, θ0, lower_bounds=0, upper_bounds=1)
    # Compare results with Python paper replication
    @test collect(θmode) ≈ r[2]
    @test r[2] ≈ [0.1789746, 0.90844006, 0.03158113] atol=1e-4
    h = logdensity_hessian!(bm, θmode)
    @test h[3] ≈ 566.4799427986145 atol=1e-3
    Σ = vcov(bm, θmode)
    # Compare results with Python paper replication
    @test Σ ≈ [1.03548913e-4 -1.50550473e-5 3.03194530e-5;
        -1.50550473e-5 7.48752696e-4 5.37343419e-5;
        3.03194530e-5 5.37343419e-5 5.50603658e-4] atol=5e-7
    se = stderror(bm, θmode)
    @test se ≈ [0.0101759, 0.02736335, 0.02346495] atol=1e-5

    @test sprint(show, bm) == "156×1 BayesianModel{Float64}(3, 0)"
    @test sprint(show, MIME("text/plain"), bm) == """
        156×1 BayesianModel{Float64} with 3 shock parameters and 0 structural parameter:
          shock parameters: σ, ar, ma"""

    bm2 = transform(as((σ=asℝ₊, ar=as𝕀, ma=as𝕀)), bm)
    θmode2, rx2, _, r2 = mode(bm2, :LD_LBFGS, θ0, lower_bounds=-5, upper_bounds=3)
    @test rx2 ≈ [-1.7205119057113951, 2.2947334490599407, -3.4231049782763914] atol=1e-5
    rr = collect(θmode2)
    @test rr ≈ rx atol=1e-6
    l2, dl2 = logdensity_and_gradient(bm2, rr)
    @test l2 ≈ -58.03849625108774 atol=1e-5
    @test dl2 ≈ [-146.3345012664795, 2.810868263244629, -3.8802170753479004] atol=1e-4
    @test vcov(parent(bm2), θmode2) ≈ Σ atol=5e-8
    @test stderror(parent(bm2), θmode2) ≈ se atol=1e-7

    @test sprint(show, bm2) == "156×1 TransformedBayesianModel(3)"
    @test sprint(show, MIME("text/plain"), bm2) == """
        156×1 TransformedBayesianModel of dimension 3 from BayesianModel{Float64} with 3 shock parameters and 0 structural parameter:
          [1:3] NamedTuple of transformations
            [1:1] :σ → asℝ₊
            [2:2] :ar → as𝕀
            [3:3] :ma → as𝕀"""

    spl = MetropolisHastings(RandomWalkProposal{true}(MvNormal(zeros(3), 2.5.*Hermitian(Σ))))
    # Small sample size to save time
    Ndrop = 3000
    N = 5000
    @time chain = sample(bm, spl, N, init_params=rx,
        param_names=collect(keys(bm[])), chain_type=Chains, progress=false)
    @test acceptance_rate(view(chain.value, Ndrop+1:N, 1, 1)) < 0.4

    tr = as((σ=as(Real,0.01,4), ar=as(Real,0.02,0.98), ma=as(Real,0.02,0.98)))
    bm3 = transform(tr, bm)
    θmode3, rx3, _, _ = mode(bm3, :LD_LBFGS, zeros(3), lower_bounds=-5, upper_bounds=5)
    @test collect(θmode3) ≈ rx atol=1e-6
    Σ3 = vcov(bm3, rx3)
    spl3 = MetropolisHastings(RandomWalkProposal{true}(MvNormal(zeros(3), 2.5.*Hermitian(Σ3))))
    @time chain3 = sample(bm3, spl3, N, init_params=rx3,
        param_names=collect(keys(bm[])), chain_type=Chains, progress=false)
    @test acceptance_rate(view(chain3.value, Ndrop+1:N, 1, 1)) < 0.2
    postmh = StructArray(transform(tr, view(chain3.value,t,1:3,1)) for t in 1:N)

    @time r = mcmc_with_warmup(Random.default_rng(), bm3, N÷10; reporter=NoProgressReport())
    posthmc = StructArray(transform(tr, v) for v in eachcol(r.posterior_matrix))
end
