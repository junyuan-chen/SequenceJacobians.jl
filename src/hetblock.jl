const Fun = Function
const FunOrN = Union{Function,Nothing}

struct HetBlock{HA,CA<:ValidCache,V<:Fun,Vss<:Fun,Vst<:FunOrN,VC<:Fun,Λ<:Fun,Λss<:Fun,
        Λst<:FunOrN,ΛC<:Fun,F<:Fun,IV<:FunOrN,IΛ<:FunOrN} <: AbstractBlock
    ins::Vector{Symbol}
    invars::Vector{VarSpec}
    ssins::Set{Symbol}
    outs::Vector{Symbol}
    cache::CA
    ha::HA
    v::V
    vss::Vss
    vssstatus::Vst
    vssconverged::VC
    λ::Λ
    λss::Λss
    λssstatus::Λst
    λssconverged::ΛC
    f::F
    initvss::IV
    initλss::IΛ
    ssargs::Dict{Symbol,Any}
    function HetBlock(ins::Vector{Symbol}, invars::Vector{VarSpec}, ssins::Set{Symbol},
            outs::Vector{Symbol}, cache::CA, ha,
            v::V, vss::Vss, vssstatus::Vst, vssconverged::VC,
            λ::Λ, λss::Λss, λssstatus::Λst, λssconverged::ΛC, f::F, initvss::IV, initλss::IΛ,
            ssargs::Dict{Symbol,Any}) where {CA,V,Vss,Vst,VC,Λ,Λss,Λst,ΛC,F,IV,IΛ}
        _checkinsouts(ins, outs, ssins)
        return new{typeof(ha),CA,V,Vss,Vst,VC,Λ,Λss,Λst,ΛC,F,IV,IΛ}(
            ins, invars, ssins, outs, cache, ha,
            v, vss, vssstatus, vssconverged, λ, λss, λssstatus, λssconverged, f,
            initvss, initλss, ssargs)
    end
end

function block(ha, v::Fun, vssconverged::Fun,
        λ::Fun, λssconverged::Fun, f::Function, ins, outs;
        ssins=ins, vss::Fun=v, vssstatus=nothing, λss::Fun=λ, λssstatus=nothing,
        initvss=nothing, initλss=nothing, ssargs=nothing, cache=nothing)
    ssargs = ssargs===nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(ssargs...)
    return HetBlock(_inout(ins, outs, ssins)..., cache, ha, v, vss, vssstatus,
        vssconverged, λ, λss, λssstatus, λssconverged, f, initvss, initλss, ssargs)
end

_verbose(verbose::Integer) = (verbose>0, Int(verbose))
_verbose(verbose::Bool) = (verbose, 20)
_verbose(verbose) = throw(ArgumentError("invalid specification of keyword verbose"))

function _backwardss(ha, f, vst::Nothing, c, invals, iter, verbose, pgap)
    verbose && iszero(iter%pgap) && println("  backward iteration $iter...")
    f(ha, invals...)
    return c(ha, nothing)
end

function _backwardss(ha, f, vst, c, invals, iter, verbose, pgap)
    verbose && iszero(iter%pgap) && println("  backward iteration $iter...")
    f(ha, invals...)
    st = vst(ha)
    verbose && println("    ", st)
    return c(ha, st)
end

function _forwardss(ha, f, λst::Nothing, c, invals, iter, verbose, pgap)
    verbose && iszero(iter%pgap) && println("  forward iteration $iter...")
    f(ha, invals...)
    return c(ha, nothing)
end

function _forwardss(ha, f, λst, c, invals, iter, verbose, pgap)
    verbose && iszero(iter%pgap) && println("  forward iteration $iter...")
    f(ha, invals...)
    st = λst(ha)
    verbose && println("    ", st)
    return c(ha, st)
end

function steadystate!(b::HetBlock, varvals::AbstractDict)
    ha = b.ha
    invals = ((varvals[n] for n in inputs(b))...,)
    verbose, pgap = _verbose(get(b.ssargs, :verbose, false))
    # Backward iterations
    b.initvss === nothing || b.initvss(ha, invals...)
    maxviter = Int(get(b.ssargs, :maxviter, 1000))
    vssconverged = false
    iter = 0
    while iter < maxviter
        iter += 1
        vssconverged = _backwardss(ha, b.vss, b.vssstatus, b.vssconverged,
            invals, iter, verbose, pgap)
        vssconverged && break
    end
    vssconverged || @warn "backward iterations did not converge after $iter steps"
    vssconverged && verbose && println("backward iteration converged after $iter steps")
    # Forward iterations
    b.initλss === nothing || b.initλss(ha, invals...)
    maxλiter = Int(get(b.ssargs, :maxλiter, 1000))
    λssconverged = false
    iter = 0
    while iter < maxλiter
        iter += 1
        λssconverged = _backwardss(ha, b.λss, b.λssstatus, b.λssconverged,
            invals, iter, verbose, pgap)
        λssconverged && break
    end
    λssconverged || @warn "forward iterations did not converge after $iter steps"
    λssconverged && verbose && println("forward iteration converged after $iter steps")
    # Obtain aggregate outcomes
    ca = b.cache
    vals = ca === nothing ? b.f(ha, invals...) : b.f(ha, ca, invals...)
    for (i, n) in enumerate(b.outs)
        val = get(varvals, n, nothing)
        val isa AbstractArray ? copyto!(val, vals[i]) : (varvals[n] = vals[i])
    end
end

