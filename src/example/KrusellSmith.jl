module KrusellSmith

using ..SequenceJacobians
using ..SequenceJacobians.RBC: firm
using ..SequenceJacobians.ExampleUtils
using LinearAlgebra: dot, mul!

import SequenceJacobians: endostates, endopolicies, exogstates, valuevars, policies, backward_init!, update!

export kshhblock, ksblocks

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
    Dtemp::Matrix{TF}
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

endostates(::KSHousehold) = (:aproc,)
endopolicies(::KSHousehold) = (aproc=:a,)
exogstates(::KSHousehold) = (:eproc,)
valuevars(::KSHousehold) = (:Va,)
policies(::KSHousehold) = (:a, :c)

function backward_init!(h::KSHousehold, r, w, β, eis)
    fill!(h.a, 0.0)
    fill!(h.c, 0.0)
    h.coh .= (1 + r) .* grid(h.aproc) .+ w .* grid(h.eproc)'
    h.Va .= (1 + r) .* (0.1 .* h.coh).^(-1/eis)
end

function update!(h::KSHousehold, r, w, β, eis)
    agrid = grid(h.aproc)
    h.cnext .= (β.*h.EVa).^(-eis)
    h.coh .= (1 + r) .* agrid .+ w .* grid(h.eproc)'
    h.cohnext .= h.cnext .+ agrid
    for i in 1:length(grid(h.eproc))
        interpolate_y!(view(h.a,:,i), view(h.coh,:,i), agrid, view(h.cohnext,:,i))
    end
    # Ensure that asset is always nonnegative
    h.a[h.a.<agrid[1]] .= agrid[1]
    h.c .= h.coh .- h.a
    h.Va .= (1 + r).*h.c.^(-1/eis)
    # It does not matter which variable is returned
    return h.Va
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

kssstarget(r, Y) = r, Y

function ksblocks(; hhkwargs...)
    bhh = kshhblock(0, 200, 500, 0.966, 0.5, 7; hhkwargs...)
    bfirm = block(firm, [lag(:K), :L, :Z, :α, :δ], [:r, :w, :Y])
    bmkt = block(mkt_clearing, [:K, :A, :Y, :C, :δ], [:asset_mkt, :goods_mkt])
    bss = block(kssstarget, [:r, :Y], [:rss, :Yss])
    return bhh, bfirm, bmkt, bss
end

end
