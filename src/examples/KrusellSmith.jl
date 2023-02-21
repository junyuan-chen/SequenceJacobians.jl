module KrusellSmith

using ..SequenceJacobians
using ..SequenceJacobians.RBC: firm_blk
using Distributions

import SequenceJacobians: endoprocs, exogprocs, valuevars, expectedvalues, policies,
    backwardtargets, backward_init!, backward_endo!

export KSHousehold, kshhblock, ksblocks, kspriors

struct KSHousehold{TF<:AbstractFloat} <: AbstractHetAgent
    aproc::EndoProc{TF,2}
    eproc::ExogProc{TF}
    c::Matrix{TF}
    cnext::Matrix{TF}
    clast::Matrix{TF}
    a::Matrix{TF}
    alast::Matrix{TF}
    coh::Matrix{TF}
    cohnext::Matrix{TF}
    Va::Matrix{TF}
    EVa::Matrix{TF}
    D::Matrix{TF}
    Dendo::Matrix{TF}
    Dlast::Matrix{TF}
end

function KSHousehold(amin, amax, Na, ρe, σe, Ne)
    aproc = assetproc(amin, amax, Na, Na, Ne)
    eproc = rouwenhorstexp(ρe, σe, Ne)
    c = Matrix{Float64}(undef, Na, Ne)
    cnext = similar(c)
    clast = similar(c)
    a = similar(c)
    alast = similar(c)
    coh = similar(c)
    cohnext = similar(c)
    Va = similar(c)
    EVa = similar(c)
    D = similar(c)
    Dtemp = similar(c)
    Dlast = similar(c)
    return KSHousehold{eltype(c)}(aproc, eproc, c, cnext, clast, a, alast,
        coh, cohnext, Va, EVa, D, Dtemp, Dlast)
end

endoprocs(h::KSHousehold) = (h.aproc,)
exogprocs(h::KSHousehold) = (h.eproc,)
valuevars(h::KSHousehold) = (h.Va,)
expectedvalues(h::KSHousehold) = (h.EVa,)
policies(h::KSHousehold) = (h.a, h.c)
backwardtargets(h::KSHousehold) = (h.a=>h.alast, h.c=>h.clast)

function backward_init!(h::KSHousehold, r, w, β, eis)
    h.coh .= (1 + r) .* grid(h.aproc) .+ w .* grid(h.eproc)'
    h.Va .= (1 + r) .* (0.1 .* h.coh).^(-1/eis)
end

function backward_endo!(h::KSHousehold, EVa, r, w, β, eis)
    agrid = grid(h.aproc)
    h.cnext .= (β.*EVa).^(-eis)
    h.coh .= (1 + r) .* agrid .+ w .* grid(h.eproc)'
    h.cohnext .= h.cnext .+ agrid
    for i in 1:length(grid(h.eproc))
        interpolate_y!(view(h.a,:,i), view(h.coh,:,i), agrid, view(h.cohnext,:,i))
    end
    # Ensure that asset is always nonnegative
    setmin!(h.a, agrid[1])
    h.c .= h.coh .- h.a
    h.Va .= (1 + r).*h.c.^(-1/eis)
end

function kshhblock(amin, amax, Na, ρe, σe, Ne; kwargs...)
    kshh = KSHousehold(amin, amax, Na, ρe, σe, Ne)
    return block(kshh, [:r, :w, :β, :eis], [:A, :C]; kwargs...)
end

function mkt_clearing(K, A, Y, C, δ)
    asset_mkt = A - K
    goods_mkt = Y - C - δ * K
    return asset_mkt, goods_mkt
end

function ksblocks(; hhkwargs...)
    bhh = kshhblock(0, 200, 500, 0.966, 0.5, 7; hhkwargs...)
    bfirm = firm_blk()
    bmkt = block(mkt_clearing, [:K, :A, :Y, :C, :δ], [:asset_mkt, :goods_mkt])
    return bhh, bfirm, bmkt
end

function kspriors()
    sh = arma11shock(:σ, :ar, :ma, :Z)
    priors = [:σ=>InverseGamma(2.01, 0.4*(2.01-1)),
        :ar=>Beta(2.625, 2.625), :ma=>Beta(2.625, 2.625)]
    return sh, priors
end

end
