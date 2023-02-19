@testset "bayesian" begin
    data = exampledata(:bayes)
    data[:,:y] ./= 4
    using SequenceJacobians.KrusellSmith
    m = model(ksblocks())
    calis = [:eis=>1, :δ=>0.025, :α=>0.11, :L=>1]
    tars = [:r=>0.01, :Y=>1, :asset_mkt=>0]
    inits = [:β=>0.98, :Z=>0.85, :K=>3]
    ss =  SteadyState(m, calis, inits, tars)
    solve!(GSL_Hybrids, ss, xtol=1e-10)
    j = TotalJacobian(m, [:Z,:K], [:asset_mkt], getvarvals(ss), 300, excluded=(:goods_mkt,))
    gj = GEJacobian(j, :Z, keepH_U=true)
    sh, priors = kspriors()
    shock = arma11shock(:σ, :ar, :ma, :Z)
    bm = bayesian(gj, shock, :Y=>:y, priors, data)
    θ = [0.2, 0.9, 0.003]
    lpri = logprior(bm, θ)
    lpost = logposterior!(bm, θ)
    # Compare results with Python paper replication
    # log prior is not comparable as the Python code does not add constant terms
    @test lpost - lpri - nrow(data)*log(2*pi)/2 ≈ 25.315994892433707 atol=1e-4
    @test logposterior!(bm, (θ...,)) ≈ lpost
    l, dl = logposterior_and_gradient!(bm, θ)
    @test l ≈ lpost
    @test dl ≈ [-166.70030975341797, 1.4917106628417969, 504.52807235717773] atol=1e-4
    bm1 = bayesian(gj, shock, :Y=>:y, priors, data, fdtype=Val(:central))
    l1, dl1 = logposterior_and_gradient!(bm1, θ)
    @test dl1 ≈ dl atol=1e-2

    @test dimension(bm) == 3
end
