abstract type AbstractHetAgent end

abstract type HetAgentStyle end

struct TimeDiscrete <: HetAgentStyle end

HetAgentStyle(ha::AbstractHetAgent) = HetAgentStyle(typeof(ha))
HetAgentStyle(::Type{<:AbstractHetAgent}) = TimeDiscrete()

"""
    endostates(ha::AbstractHetAgent)

Return an iterable object that contains all names used for
identifying each endogenous law of motion of `ha`.
Each element of the returned object must be accepted by [`getendo`](@ref).
"""
function endostates end

"""
    getendo(ha::AbstractHetAgent, n::Symbol)

Return the object that holds the endogenous law of motion `n` from `ha`.
The fallback method assumes that `n` is a property of `ha`
and returns this property.
"""
getendo(ha::AbstractHetAgent, n::Symbol) = getproperty(ha, n)

@inline function endoprocs(ha::AbstractHetAgent)
    endos = endostates(ha)
    return ntuple(i->@inbounds(getendo(ha, endos[i])), length(endos))
end

"""
    endopolicies(ha::AbstractHetAgent)

Return an object that maps each endogenous state to the corresponding policy.
The policy can be retrieved by `getproperty(ha, n)` with `n` being the name of the state.
"""
function endopolicies end

"""
    exogstates(ha::AbstractHetAgent)

Return an iterable object that contains all names used for
identifying each exogenous law of motion of `ha`.
Each element of the returned object must be accepted by [`getexog`](@ref).
"""
function exogstates end

"""
    getexog(ha::AbstractHetAgent, n::Symbol)

Return the object that holds the exogenous law of motion `n` from `ha`.
The fallback method assumes that `n` is a property of `ha`
and returns this property.
"""
getexog(ha::AbstractHetAgent, n::Symbol) = getproperty(ha, n)

@inline function exogprocs(ha::AbstractHetAgent)
    exogs = exogstates(ha)
    return ntuple(i->@inbounds(getexog(ha, exogs[i])), length(exogs))
end

"""
    statevars(ha::AbstractHetAgent)

Return an iterable object that contains all names used for
identifying each state variable of `ha`.
The order is consistent with the axes of the grids.
This method is based on [`endostates`](@ref) and [`exogstates`](@ref).
"""
statevars(ha::AbstractHetAgent) = (endostates(ha)..., exogstates(ha)...)

"""
    valuevars(ha::AbstractHetAgent)

Return an iterable object that contains all names used for
identifying the value function or its partial derivatives
involved in the backward iteration of `ha`.
Each element of the returned object must be accepted by
[`getvalue`](@ref) and [`getexpectedvalue`](@ref).
"""
function valuevars end

"""
    getvalue(ha::AbstractHetAgent, n::Symbol)

Return the object that holds the value `n` associated with
the current step of backward iteration from `ha`.
The fallback method assumes that `n` is a property of `ha`
and returns this property.
See also [`getexpectedvalue`](@ref).
"""
getvalue(ha::AbstractHetAgent, n::Symbol) = getproperty(ha, n)

@inline function getvalues(ha::AbstractHetAgent)
    vars = valuevars(ha)
    return ntuple(i->@inbounds(getvalue(ha, vars[i])), length(vars))
end

"""
    getexpectedvalue(ha::AbstractHetAgent, n::Symbol)

Return the object that holds the value `n` associated with
the previous step of backward iteration from `ha`.
The fallback method assumes that `Symbol(:E, n)` is a property of `ha`
and returns this property.
See also [`getvalue`](@ref).
"""
getexpectedvalue(ha::AbstractHetAgent, n::Symbol) = getproperty(ha, Symbol(:E, n))

@inline function expectedvalues(ha::AbstractHetAgent)
    vars = valuevars(ha)
    return ntuple(i->@inbounds(getexpectedvalue(ha, vars[i])), length(vars))
end

"""
    policies(ha::AbstractHetAgent)

Return an iterable object that contains all names used for
identifying each policy of `ha`.
The returned object must contain names of the policies
associated with the endogenous states
and places them in the beginning in the same order as
how the states are indexed by the object returned by [`endostates`](@ref).
Each element of the returned object must be accepted by
[`getpolicy`](@ref) and [`getlastpolicy`](@ref).
"""
function policies end

"""
    getpolicy(ha::AbstractHetAgent, n::Symbol)

Return the object that holds the policy `n` associated with
the current step of backward iteration from `ha`.
The fallback method assumes that `n` is a property of `ha`
and returns this property.
See also [`getlastpolicy`](@ref).
"""
getpolicy(ha::AbstractHetAgent, n::Symbol) = getproperty(ha, n)

@inline function getpolicies(ha::AbstractHetAgent)
    pols = policies(ha)
    return ntuple(i->@inbounds(getpolicy(ha, pols[i])), length(pols))
end

"""
    getlastpolicy(ha::AbstractHetAgent, n::Symbol)

Return the object that holds the policy `n` associated with
the previous step of backward iteration from `ha`.
The fallback method assumes that `Symbol(n, :last)` is a property of `ha`
and returns this property.
See also [`getpolicy`](@ref).
"""
getlastpolicy(ha::AbstractHetAgent, n::Symbol) = getproperty(ha, Symbol(n, :last))

"""
    getdist(ha::AbstractHetAgent)

Return the distribution of agents associated with
the current step of forward iteration from `ha`.
The fallback method assumes that `D` is a property of `ha`
and returns this property.
See also [`getlastdist`](@ref) and [`getdistendo`](@ref).
"""
getdist(ha::AbstractHetAgent) = getproperty(ha, :D)

"""
    getlastdist(ha::AbstractHetAgent)

Return the distribution of agents associated with
the previous step of forward iteration from `ha`.
The fallback method assumes that `Dlast` is a property of `ha`
and returns this property.
See also [`getdist`](@ref) and [`getdistendo`](@ref).
"""
getlastdist(ha::AbstractHetAgent) = getproperty(ha, :Dlast)

"""
    getdistendo(ha::AbstractHetAgent)

Return the distribution of agents after the transition
driven by the endogenous law of motion but before being hitted by exogenous shocks.
The fallback method assumes that `Dendo` is a property of `ha`
and returns this property.
See also [`getdist`](@ref) and [`getlastdist`](@ref).
"""
getdistendo(ha::AbstractHetAgent) = getproperty(ha, :Dendo)

"""
    backward_exog!(ha::AbstractHetAgent)

Compute the expected values given the current values and the law of motion of exogenous states.
"""
function backward_exog!(ha::AbstractHetAgent)
    exogs = exogprocs(ha)
    for n in valuevars(ha)
        v = getvalue(ha, n)
        ev = getexpectedvalue(ha, n)
        backward!(ev, v, exogs...)
    end
end

"""
    backward_endo!(ha::AbstractHetAgent, EVs..., invals...)

Update the values and policies of `ha` given the expected values `EVs`
and macro variables evaluated at `invals`.
To allow the computation of Jacobians,
reference to the arrays of expected values must be done via `EVs`
instead of any array contained in `ha`.
This method is essential for computing the sequence-space Jacobians and transitional paths.
"""
function backward_endo! end

"""
    backward!(ha::AbstractHetAgent, invals...)

Iterate the values of `ha` backward by one step with macro variables evaluated at `invals`.
A fallback method is selected based on [`HetAgentStyle`](@ref).
"""
backward!(ha::AbstractHetAgent, invals...) = backward!(HetAgentStyle(ha), ha, invals...)

function backward!(::TimeDiscrete, ha::AbstractHetAgent, invals...)
    for n in policies(ha)
        copyto!(getlastpolicy(ha, n), getpolicy(ha, n))
    end
    backward_exog!(ha)
    backward_endo!(ha, expectedvalues(ha)..., invals...)
end

"""
    backward_steadystate!(ha::AbstractHetAgent, invals...)

A variant of [`backward!`](@ref) used when solving the steady state.
A fallback method is selected based on [`HetAgentStyle`](@ref)
and may simply call [`backward!`](@ref).
"""
backward_steadystate!(ha::AbstractHetAgent, invals...) =
    backward!(HetAgentStyle(ha), ha, invals...)

backward_steadystate!(hs::TimeDiscrete, ha, invals...) = backward!(hs, ha, invals...)

"""
    backward_init!(ha::AbstractHetAgent, invals...)

Initialize data objects contained in `ha` before backward iteration.
The fallback method returns `nothing` without making any change.
"""
backward_init!(::AbstractHetAgent, invals...) = nothing

"""
    backward_status(ha::AbstractHetAgent)

Return an object that indicates the status of backward iteration based on values in `ha`.
This method allows tracing the steps of backward iteration for inspection.
The fallback method returns `nothing`.
"""
backward_status(::AbstractHetAgent) = nothing

"""
    backward_converged(ha::AbstractHetAgent, status, tol::Real=1e-8)

Assess whether the backward iteration has converged at tolerance level `tol`
based on values in `ha` and `status` returned by [`backward_status`](@ref).
The fallback method returns `true` if [`supconverged`](@ref) returns `true`
for all pairs of current and last policies while disregarding `status`.
"""
backward_converged(ha::AbstractHetAgent, st, tol::Real=1e-8) =
    all(n->supconverged(getpolicy(ha, n), getlastpolicy(ha, n), tol), policies(ha))

"""
    forward!(ha::AbstractHetAgent, invals...)

Iterate the distributions of `ha` forward by one step
with macro variables evaluated at `invals`.
Unless in special circumstances,
the fallback method selected based on [`HetAgentStyle`](@ref) should be sufficient
and there is no need to add methods to this function.
"""
forward!(ha::AbstractHetAgent, invals...) = forward!(HetAgentStyle(ha), ha, invals...)

function forward!(::TimeDiscrete, ha::AbstractHetAgent, invals...)
    D = getdist(ha)
    Dlast = getlastdist(ha)
    Dendo = getdistendo(ha)
    copyto!(Dlast, D)
    forward!(Dendo, Dlast, endoprocs(ha)...)
    forward!(D, Dendo, exogprocs(ha)...)
end

"""
    forward_steadystate!(ha::AbstractHetAgent, invals...)

A variant of [`forward!`](@ref) used when solving the steady state.
A fallback method is selected based on [`HetAgentStyle`](@ref)
and may simply call [`forward!`](@ref).
"""
forward_steadystate!(ha::AbstractHetAgent, invals...) =
    forward!(HetAgentStyle(ha), ha, invals...)

forward_steadystate!(hs::TimeDiscrete, ha, invals...) = forward!(hs, ha, invals...)

"""
    forward_init!(ha::AbstractHetAgent, invals...)

Initialize data objects contained in `ha` before forward iteration.
A fallback method is selected based on [`HetAgentStyle`](@ref)
and should be sufficient in most scenarios.
"""
forward_init!(ha::AbstractHetAgent, invals...) =
    forward_init!(HetAgentStyle(ha), ha, invals...)

function _initdist!(D::AbstractArray, ds::Vararg{Vector,N}) where N
    nD = ndims(D)
    p0 = 1/prod(i->size(D,i), 1:nD-N)
    dims = (nD-N+1:nD...,)
    vs = splitdimsview(D, dims)
    @inbounds for (i, p) in enumerate(Base.product(ds...))
        fill!(vs[i], *(p0, p...))
    end
end

function initdist!(ha::AbstractHetAgent)
    D = getdist(ha)
    exogs = exogstates(ha)
    Nexog = length(exogs)
    if Nexog > 0
        ds = ntuple(i->getexog(ha, exogs[i]).d, Nexog)
        _initdist!(D, ds...)
    else
        fill!(D, 1/length(D))
    end
end

function initendo!(ha::AbstractHetAgent)
    endopols = endopolicies(ha)
    for (i, endo) in enumerate(endostates(ha))
        update!(getendo(ha, endo), i, getpolicy(ha, getproperty(endopols, endo)))
    end
end

function forward_init!(::TimeDiscrete, ha::AbstractHetAgent, invals...)
    initdist!(ha)
    initendo!(ha)
end

"""
    forward_status(ha::AbstractHetAgent)

Return an object that indicates the status of forward iteration based on values in `ha`.
This method allows tracing the steps of forward iteration for inspection.
The fallback method returns `nothing`.
"""
forward_status(::AbstractHetAgent) = nothing

"""
    forward_converged(ha::AbstractHetAgent, status, tol::Real=1e-8)

Assess whether the forward iteration has converged at tolerance level `tol`
based on values in `ha` and `status` returned by [`forward_status`](@ref).
The fallback method returns `true` if [`supconverged`](@ref) returns `true`
for the current and last distributions while disregarding `status`.
"""
forward_converged(ha::AbstractHetAgent, st, tol::Real=1e-8) =
    supconverged(getdist(ha), getlastdist(ha), tol)

"""
    aggregate(ha::AbstractHetAgent, invals...)

Return aggregated outcomes from each policy
based on the current distribution of agents.
The fallback method takes the dot products disregarding `invals`.
"""
function aggregate(ha::AbstractHetAgent, invals...)
    pols = policies(ha)
    D = getdist(ha)
    N = length(D)
    s2 = stride1(D)
    function _agg(i)
        pol = getpolicy(ha, pols[i])
        s1 = stride1(pol)
        return BLAS.dot(N, pol, s1, D, s2)
    end
    return ntuple(_agg, length(pols))
end
