struct MinimumDistance{TF, NT, G<:GMaps{TF}, P<:GEJacobianUpdatePlan,
        BV<:PseudoBlockVector, F<:Function, GC<:GradientCache, JC<:JacobianCache,
        FDA<:NamedTuple} <: AbstractEstimator{TF,NT}
    gs::G
    paravals::RefValue{NT}
    vars::Vector{Pair{Symbol,Vector{Symbol}}}
    plan::P
    jacmaps::Vector{AbstractJacobianMap{TF}}
    vals::BV
    tars::Vector{TF}
    weights::Vector{TF}
    f::F
    dobj::Vector{TF}
    dobjcache::GC
    x::Vector{TF}
    fx::Vector{TF}
    jac::Matrix{TF}
    jaccache::JC
    fdkwargs::FDA
end

function MinimumDistance(gs::GMaps{TF}, params, vars::Vector{Pair{Symbol,Vector{Symbol}}},
        values::PseudoBlockVector, targets, weights, f::Function=identity;
        fdtype=Val(:forward), fdkwargs=NamedTuple()) where TF
    params isa Symbol && (params = (params,))
    params = ntuple(i->params[i], length(params))
    p = plan(gs.gj, params)
    jacmaps = AbstractJacobianMap{TF}[]
    for (exo, endos) in vars
        dmaps = gs[exo]
        for endo in endos
            push!(jacmaps, dmaps[endo])
        end
    end
    length(jacmaps) > blocksize(values, 1) && throw(DimensionMismatch(
        "number of blocks in values is expected to be at least $(length(jacmaps))"))
    paravals = NamedTuple{params}(gs.gj.tjac.varvals[])
    npara = _getvarlength(params, paravals)
    ntar = length(targets)
    ntar == length(values) == length(weights) || throw(DimensionMismatch(
        "values, targets, and weights must have the same length"))
    dobj = Vector{TF}(undef, npara)
    dobjcache = GradientCache{TF,Nothing,Nothing,Vector{TF},fdtype,TF,Val(true)}(
        zero(TF), nothing, nothing, similar(dobj))
    jac = Matrix{TF}(undef, ntar, npara)
    jaccache = JacobianCache(Vector{TF}(undef, npara), Vector{TF}(undef, ntar))
    return MinimumDistance(gs, Ref(paravals), vars, p, jacmaps, copy(values),
        collect(targets), collect(weights), f, dobj, dobjcache,
        collect(TF, paravals), copy(values.blocks), jac, jaccache, fdkwargs)
end

@inline getindex(md::MinimumDistance) = md.paravals[]
@inline getindex(md::MinimumDistance, i) = getindex(md[], i)

function evaluate!(md::MinimumDistance, y, θ)
    _update_paravals!(md, θ)
    tjac = md.gs.gj.tjac
    tjac.varvals[] = varvals = merge(tjac.varvals[], md.paravals[])
    md.plan(varvals)
    md.gs()
    for (i, m) in enumerate(md.jacmaps)
        mul!(view(md.vals, Block(i)), m, true)
    end
    md.f(md.vals)
    length(y) == length(md.vals) && copyto!(y, md.vals.blocks)
    return md.vals
end

function (md::MinimumDistance)(θ)
    evaluate!(md, (), θ)
    md.fx .= (md.vals.blocks .- md.tars).^2 ./ md.weights
    return sum(md.fx)
end

function (md::MinimumDistance)(θ, grad::AbstractVector)
    c = md(θ)
    if length(grad) > 0
        ca = _update_fdcache(md.dobjcache, c)
        finite_difference_gradient!(grad, md, θ, ca; md.fdkwargs...)
    end
    return c
end

function vcov(md::MinimumDistance)
    copyto!(md.x, Iterators.flatten(md.paravals[]))
    # fx is also used as a cache for evaluating the distance
    # Assume values in md.vals.blocks match md.x
    copyto!(md.fx, md.vals.blocks)
    f(y, x) = evaluate!(md, y, x)
    finite_difference_jacobian!(md.jac, f, md.x, md.jaccache, md.fx)
    return inv!(cholesky!(Hermitian(md.jac' * Diagonal(1.0./md.weights) * md.jac)))
end

function stderror(md::MinimumDistance)
    Σ = vcov(md)
    return map(i->@inbounds(sqrt(Σ[i])), diagind(Σ))
end

show(io::IO, md::MinimumDistance{TF}) where TF = print(io, length(md.tars),
    '×', length(md.x), " MinimumDistance{", TF, "}(", length(md.vars), ')')

function show(io::IO, ::MIME"text/plain", md::MinimumDistance{TF,NT}) where {TF,NT}
    nexo = length(md.vars)
    npara = length(md.x)
    print(io, length(md.tars), '×', npara, " MinimumDistance{", TF, "} with ")
    println(io, nexo, " exogenous variable", nexo > 1 ? "s:" : ':')
    print(io, "  parameter")
    npara > 1 && print(io, 's')
    print(io, ": ")
    join(io, NT.parameters[1], ", ")
end
