struct HetBlock{HA<:AbstractHetAgent} <: AbstractBlock
    ins::Vector{Symbol}
    invars::Vector{VarSpec}
    ssins::Set{Symbol}
    outs::Vector{Symbol}
    ha::HA
    ssargs::Dict{Symbol,Any}
    function HetBlock(ins::Vector{Symbol}, invars::Vector{VarSpec}, ssins::Set{Symbol},
            outs::Vector{Symbol}, ha::HA, ssargs::Dict{Symbol,Any}) where HA
        _checkinsouts(ins, outs, ssins)
        return new{HA}(ins, invars, ssins, outs, ha, ssargs)
    end
end

function block(ha::AbstractHetAgent, ins, outs; ssins=ins, ssargs=nothing)
    ssargs = ssargs === nothing ? Dict{Symbol,Any}() : Dict{Symbol,Any}(ssargs...)
    return HetBlock(_inout(ins, outs, ssins)..., ha, ssargs)
end

_verbose(verbose::Integer) = (verbose>0, Int(verbose))
_verbose(verbose::Bool) = (verbose, 20)
_verbose(verbose) = throw(ArgumentError("invalid specification of keyword verbose"))

function _backwardss!(ha, invals, maxbackiter, backtol, verbose, pgap)
    iter = 0
    backconverged = false
    while iter < maxbackiter
        iter += 1
        verbose && iszero(iter%pgap) && println("  backward iteration $iter...")
        backward!(ha, invals...)
        st = backward_status(ha)
        st === nothing || verbose && println("    ", st)
        backconverged = backward_converged(ha, st, backtol)
        backconverged && break
    end
    backconverged ||
        @warn "backward iterations did not converge with tolerance $backtol after $iter steps"
    backconverged && verbose &&
        println("backward iteration converged with tolerance $backtol after $iter steps")
end

function _forwardss!(ha, invals, maxforiter, fortol, verbose, pgap)
    forconverged = false
    iter = 0
    while iter < maxforiter
        iter += 1
        verbose && iszero(iter%pgap) && println("  forward iteration $iter...")
        forward!(ha, invals...)
        st = forward_status(ha)
        st === nothing || verbose && println("    ", st)
        forconverged = forward_converged(ha, st, fortol)
        forconverged && break
    end
    forconverged ||
        @warn "forward iterations did not converge with tolerance $fortol after $iter steps"
    forconverged && verbose &&
        println("forward iteration converged with tolerance $fortol after $iter steps")
end

function steadystate!(b::HetBlock, varvals::AbstractDict)
    ha = b.ha
    ins = inputs(b)
    invals = ntuple(i->varvals[ins[i]], length(ins))
    verbose, pgap = _verbose(get(b.ssargs, :verbose, false))
    # Backward iterations
    backward_init!(ha, invals...)
    maxbackiter = get(b.ssargs, :maxbackiter, 1000)
    backtol = get(b.ssargs, :backtol, 1e-8)
    _backwardss!(ha, invals, maxbackiter, backtol, verbose, pgap)
    # Forward iterations
    forward_init!(ha, invals...)
    maxforiter = get(b.ssargs, :maxforiter, 1000)
    fortol = get(b.ssargs, :fortol, 1e-10)
    _forwardss!(ha, invals, maxforiter, fortol, verbose, pgap)
    # Obtain aggregate outcomes
    vals = aggregate(ha, invals...)
    for (i, n) in enumerate(outputs(b))
        val = get(varvals, n, nothing)
        val isa AbstractArray ? copyto!(val, vals[i]) : (varvals[n] = vals[i])
    end
end

