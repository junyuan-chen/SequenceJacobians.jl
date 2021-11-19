function _transform!(d::IdDict, trans::Vector{Symbol}, gejac::GEJacobian)
    varvals = gejac.jacs.varvals
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

function _transform!(d::IdDict, trans::Bool, gejac::GEJacobian)
    if trans
        varvals = gejac.jacs.varvals
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
    jacs = gejac.jacs
    dshocks isa Pair && (dshocks = (dshocks,))
    endovars isa Symbol && (endovars = (endovars,))
    endovars === nothing && (endovars = setdiff(jacs.vars, gejac.exovars, jacs.tars))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    nT = jacs.nT
    out = IdDict{Symbol,IdDict{Symbol,VecOrMat{TF}}}()
    for (exo, dZ) in dshocks
        size(dZ,1) == nT || throw(ArgumentError("the length of shock $exo is not $nT"))
        d = IdDict{Symbol,VecOrMat{TF}}()
        out[exo] = d
        for endo in endovars
            G = getG!(gejac, exo, endo)
            d[endo] = G * dZ
        end
    end
    _transform!(out, transform, gejac)
    return out
end

function linirf(jacs::TotalJacobian, dshocks::ValidPathInput, endovars=nothing;
        transform=false, keepH_U::Bool=false)
    dshocks isa Pair && (dshocks = (dshocks,))
    exovars = (exo for (exo, _) in dshocks)
    gejac = GEJacobian(jacs, exovars; keepH_U=keepH_U)
    irfs = linirf(gejac, dshocks isa Tuple ? dshocks[1] : dshocks, endovars; transform=transform)
    return irfs, gejac
end

function _transform!(d::IdDict, trans::Vector{Symbol}, jacs::TotalJacobian)
    varvals = jacs.varvals
    for v in trans
        irf = get(d, v, nothing)
        if irf !== nothing
            ss = varvals[v]
            irf .= 100.0.*irf./ss .- 100.0
        end
    end
end

function _transform!(d::IdDict, trans::Bool, jacs::TotalJacobian)
    if trans
        varvals = jacs.varvals
        for (v, irf) in d
            ss = varvals[v]
            irf .= 100.0.*irf./ss .- 100.0
        end
    end
end

function nlirf(tr::Transition{TF}, endovars=nothing; transform=false) where TF
    jacs = tr.jacs
    endovars isa Symbol && (endovars = (endovars,))
    endovars === nothing && (endovars = setdiff(jacs.vars, tr.exovars, jacs.tars))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    out = IdDict{Symbol,VecOrMat{TF}}()
    for endo in endovars
        path = get(tr.varpaths, endo, nothing)
        path === nothing && throw(ArgumentError("$endo is not an endogenous variable"))
        out[endo] = path[2]
    end
    _transform!(out, transform, jacs)
    return out
end

function nlirf(jacs::TotalJacobian, shocks::ValidPathInput, endovars=nothing;
        transform=false, initials=nothing, H_U=nothing, kwargs...)
    tr = Transition(jacs, shocks, initials; H_U=H_U)
    solve!(tr, kwargs...)
    return nlirf(tr, endovars; transform=transform), tr
end
