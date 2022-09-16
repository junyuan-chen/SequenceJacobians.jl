# Methods for linirf
function _transform!(d::Dict, trans::Vector{Symbol}, GJ::GEJacobian)
    varvals = GJ.tjac.varvals
    for irfs in values(d)
        for v in trans
            irf = get(irfs, v, nothing)
            if irf !== nothing
                ss = varvals[v]
                ss isa Array && (ss = reshape(ss,1,length(ss)))
                irf .= 100.0.*irf./ss
            end
        end
    end
end

function _transform!(d::Dict, trans::Bool, GJ::GEJacobian)
    if trans
        varvals = GJ.tjac.varvals
        for irfs in values(d)
            for (v, irf) in irfs
                ss = varvals[v]
                ss isa Array && (ss = reshape(ss,1,length(ss)))
                irf .= 100.0.*irf./ss
            end
        end
    end
end

function linirf(GJ::GEJacobian{TF}, dshocks::ValidPathInput, endovars=nothing;
        transform=false) where TF
    tjac = GJ.tjac
    dshocks isa Pair && (dshocks = (dshocks,))
    endovars isa Symbol && (endovars = (endovars,))
    endovars === nothing && (endovars = setdiff(tjac.vars, GJ.exovars, tjac.tars))
    isempty(endovars) && throw(ArgumentError("endovars cannot be empty"))
    nT = GJ.nTfull
    out = Dict{Symbol,Dict{Symbol,VecOrMat{TF}}}()
    for (exo, dZ) in dshocks
        size(dZ,1) == nT || throw(ArgumentError("the length of shock $exo is not $nT"))
        d = Dict{Symbol,VecOrMat{TF}}()
        out[exo] = d
        for endo in endovars
            M = getM!(GJ, exo, endo)
            dZ isa Vector || (dZ = view(dZ,:))
            r = M * dZ
            length(r) == nT || (r = reshape(r, nT, length(r)÷nT))
            d[endo] = r
        end
    end
    _transform!(out, transform, GJ)
    return out
end

function linirf(tjac::TotalJacobian, dshocks::ValidPathInput, endovars=nothing;
        transform=false, keepH_U::Bool=false)
    dshocks isa Pair && (dshocks = (dshocks,))
    exovars = (exo for (exo, _) in dshocks)
    GJ = GEJacobian(tjac, exovars; keepH_U=keepH_U)
    irfs = linirf(GJ, dshocks isa Tuple ? dshocks[1] : dshocks, endovars; transform=transform)
    return irfs, GJ
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
