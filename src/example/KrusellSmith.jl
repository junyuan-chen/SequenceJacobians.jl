module KrusellSmith

using ..SequenceJacobians
using ..SequenceJacobians.RBC: firm
using ..SequenceJacobians.ExampleUtils
using LinearAlgebra: dot, mul!

export kshhblock, ksblocks

struct KSHousehold{TF<:AbstractFloat}
    agrid::Vector{TF}
    egrid::Vector{TF}
    epr::Vector{TF}
    ePi::Matrix{TF}
    c::Matrix{TF}
    cnext::Matrix{TF}
    cold::Matrix{TF}
    a::Matrix{TF}
    aold::Matrix{TF}
    ai::Matrix{Int}
    api::Matrix{TF}
    coh::Matrix{TF}
    cohnext::Matrix{TF}
    Va::Matrix{TF}
    Va_p::Matrix{TF}
    D::Matrix{TF}
    Dnew::Matrix{TF}
    Dold::Matrix{TF}
end

function KSHousehold(amin, amax, Na, ρe, σe, Ne)
    agrid = grida(amax, Na, amin)
    egrid, epr, ePi = gridrouwenhorst(ρe, σe, Ne)
    c = Matrix{Float64}(undef, Na, Ne)
    cnext = similar(c)
    cold = similar(c)
    a = zeros(Na, Ne)
    aold = similar(c)
    ai = Matrix{Int}(undef, Na, Ne)
    api = similar(c)
    coh = similar(c)
    cohnext = similar(c)
    Va = similar(c)
    Va_p = similar(c)
    D = similar(c)
    Dnew = similar(c)
    Dold = similar(c)
    return KSHousehold{eltype(c)}(agrid, egrid, epr, ePi, c, cnext, cold, a, aold,
        ai, api, coh, cohnext, Va, Va_p, D, Dnew, Dold)
end

function backward_init!(h::KSHousehold, r, w, β, eis)
    fill!(h.a, 0.0)
    fill!(h.c, 0.0)
    h.coh .= (1+r).*h.agrid .+ w.*h.egrid'
    h.Va .= (1+r).*(0.1.*h.coh).^(-1/eis)
    # It does not matter which variable is returned
    return h.Va
end

function household!(h::KSHousehold, r, w, β, eis)
    # Copy policies from the last iteration for assessing convergence
    copyto!(h.aold, h.a)
    copyto!(h.cold, h.c)
    # Update expected values
    mul!(h.Va_p, h.Va, h.ePi')
    h.cnext .= (β.*h.Va_p).^(-eis)
    h.coh .= (1+r).*h.agrid .+ w.*h.egrid'
    h.cohnext .= h.cnext .+ h.agrid
    for i in 1:length(h.egrid)
        interpolate_y!(view(h.a,:,i), view(h.coh,:,i), h.agrid, view(h.cohnext,:,i))
    end
    # Ensure that asset is always nonnegative
    h.a[h.a.<h.agrid[1]] .= h.agrid[1]
    h.c .= h.coh .- h.a
    h.Va .= (1+r).*h.c.^(-1/eis)
    # It does not matter which variable is returned
    return h.Va
end

policyconverged(h::KSHousehold, st) =
    linfconverged(h.a, h.aold, 1e-8) && linfconverged(h.c, h.cold, 1e-8)

function forward_init!(h::KSHousehold, r, w, β, eis)
    for i in 1:length(h.epr)
        fill!(view(h.D,:,i), h.epr[i]/length(h.agrid))
        interpolate_coord!(view(h.ai,:,i), view(h.api,:,i), view(h.a,:,i), h.agrid)
    end
    # It does not matter which variable is returned
    return h.D
end

function hhforward!(h::KSHousehold, r, w, β, eis)
    copyto!(h.Dold, h.D)
    forward_policy_1d!(h.Dnew, h.D, h.ai, h.api)
    mul!(h.D, h.Dnew, h.ePi)
end

dconverged(h::KSHousehold, st) = linfconverged(h.D, h.Dold, 1e-10)

function hhagg(h::KSHousehold, r, w, β, eis)
    A = dot(h.a, h.D)
    C = dot(h.c, h.D)
    return A, C
end

function kshhblock(amin, amax, Na, ρe, σe, Ne; kwargs...)
    kshh = KSHousehold(amin, amax, Na, ρe, σe, Ne)
    return block(kshh, household!, policyconverged, hhforward!, dconverged, hhagg,
        [:r, :w, :β, :eis], [:A, :C]; initvss=backward_init!, initλss=forward_init!, kwargs...)
end

function mkt_clearing(K, A, Y, C, δ)
    asset_mkt = A - K
    goods_mkt = Y - C - δ * K
    return asset_mkt, goods_mkt
end

kssstarget(r, Y) = r, Y

function ksblocks(; hhkwargs...)
    bhh = kshhblock(0, 200, 500, 0.966, 0.5, 7; hhkwargs...)
    bfirm = block(firm, [lag(:K), :L, :Z, :α, :δ], [:r, :w, :Y])
    bmkt = block(mkt_clearing, [:K, :A, :Y, :C, :δ], [:asset_mkt, :goods_mkt])
    bss = block(kssstarget, [:r, :Y], [:rss, :Yss])
    return bhh, bfirm, bmkt, bss
end

end
