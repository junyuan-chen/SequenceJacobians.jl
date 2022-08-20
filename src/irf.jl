# Methods for linirf
function _transform!(d::Dict, trans::Vector{Symbol}, gejac::GEJacobian)
    varvals = gejac.tjac.varvals
    for irfs in values(d)
        for v in trans
            irf = get(irfs, v, nothing)
            if irf !== nothing
                ss = varvals[v]
                irf .= 100.0.*irf./ss
            end
        end
    end
end

function _transform!(d::Dict, trans::Bool, gejac::GEJacobian)
    if trans
        varvals = gejac.tjac.varvals
        for irfs in values(d)
            for (v, irf) in irfs
                ss = varvals[v]
                irf .= 100.0.*irf./ss
            end
        end
    end
end

function linirf(gejac::GEJacobian{TF}, dshocks::ValidPathInput, endovars=nothing;
        transform=false) where TF
    tjac = gejac.tjac
    dshocks isa Pair && (dshocks = (dshocks,))
    endovars isa Symbol && (endovars = (endovars,))
    endovars === nothing && (endovars = setdiff(tjac.vars, gejac.exovars, tjac.tars))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    nT = tjac.nT
    out = Dict{Symbol,Dict{Symbol,VecOrMat{TF}}}()
    for (exo, dZ) in dshocks
        size(dZ,1) == nT || throw(ArgumentError("the length of shock $exo is not $nT"))
        d = Dict{Symbol,VecOrMat{TF}}()
        out[exo] = d
        for endo in endovars
            G = getG!(gejac, exo, endo)
            d[endo] = G * dZ
        end
    end
    _transform!(out, transform, gejac)
    return out
end

function linirf(tjac::TotalJacobian, dshocks::ValidPathInput, endovars=nothing;
        transform=false, keepH_U::Bool=false)
    dshocks isa Pair && (dshocks = (dshocks,))
    exovars = (exo for (exo, _) in dshocks)
    gejac = GEJacobian(tjac, exovars; keepH_U=keepH_U)
    irfs = linirf(gejac, dshocks isa Tuple ? dshocks[1] : dshocks, endovars; transform=transform)
    return irfs, gejac
end

# Methods for nlirf
function _transform!(d::Dict, trans::Vector{Symbol}, tjac::TotalJacobian)
    varvals = tjac.varvals
    for v in trans
        irf = get(d, v, nothing)
        if irf !== nothing
            ss = varvals[v]
            d[v] = 100.0.*(irf./ss .- 1.0)
        end
    end
end

function _transform!(d::Dict, trans::Bool, tjac::TotalJacobian)
    if trans
        varvals = tjac.varvals
        for (v, irf) in d
            ss = varvals[v]
            d[v] = 100.0.*(irf./ss .- 1.0)
        end
    end
end

function nlirf(tr::Transition{TF}, endovars=nothing; transform=false) where TF
    tjac = tr.tjac
    endovars isa Symbol && (endovars = (endovars,))
    endovars === nothing && (endovars = setdiff(tjac.vars, tr.exovars, tjac.tars))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    out = Dict{Symbol,VecOrMat{TF}}()
    for endo in endovars
        path = get(tr.varpaths, endo, nothing)
        path === nothing && throw(ArgumentError("$endo is not an endogenous variable"))
        out[endo] = path[2]
    end
    _transform!(out, transform, tjac)
    return out
end

function nlirf(tjac::TotalJacobian, shocks::ValidPathInput, endovars=nothing;
        transform=false, initials=nothing, H_U=nothing, kwargs...)
    tr = Transition(tjac, shocks, initials; H_U=H_U)
    solve!(tr; kwargs...)
    return nlirf(tr, endovars; transform=transform), tr
end

astable(irfs::Dict{Symbol}) = (; (s=>(; irfs[s]...) for s in keys(irfs))...)
