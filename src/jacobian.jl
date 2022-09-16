const JacMap{TF} = Union{LinearMap{TF}, Matrix{T}} where T<:LinearMap{TF}

struct TotalJacobian{TF<:AbstractFloat,NT<:NamedTuple}
    parent::SequenceSpaceModel
    blks::Vector{AbstractBlock}
    vars::Set{Symbol}
    varvals::NT
    nT::Int
    srcs::Set{Symbol}
    tars::Vector{Symbol}
    ntarsrc::Vector{Int}
    excluded::Union{Set{BlockOrVar},Nothing}
    parts::Dict{Symbol,Dict{Symbol,Union{Matrix{<:LinearMap{TF}},MatMulMap{TF}}}}
    totals::Dict{Symbol,Dict{Symbol,JacMap{TF}}}
end

function TotalJacobian(m::SequenceSpaceModel, sources, targets,
        varvals::NamedTuple, nT::Int; excluded=nothing, TF::Type=Float64)
    sources isa Symbol && (sources = (sources,))
    targets isa Symbol && (targets = (targets,))
    excluded isa BlockOrVar && (excluded = (excluded,))
    isempty(sources) && throw(ArgumentError("sources cannot be empty"))
    isempty(targets) && throw(ArgumentError("targets cannot be empty"))
    sources = Set{Symbol}(sources)
    # The order of the targets need to be maintained
    targets = collect(Symbol, targets)
    any(v->v in sources, targets) && throw(ArgumentError("sources and targets cannot overlap"))
    excluded === nothing || (excluded = Set{BlockOrVar}(excluded))
    vars = copy(sources)
    blks = AbstractBlock[]
    DMat = Dict{Symbol, Union{Matrix{<:LinearMap{TF}},MatMulMap{TF}}}
    parts = Dict{Symbol, DMat}()
    DMap = Dict{Symbol, JacMap{TF}}
    totals = Dict{Symbol, DMap}(u=>DMap() for u in vars)
    zmap = LinearMap(UniformScaling(zero(TF)), nT)
    for v in m.order
        isblock(m, v) || continue
        blk = m.pool[v]
        excluded !== nothing && blk in excluded && continue
        pushed = false
        # For some blocks, Jacobians for all inputs are obtained together
        byinput = jacbyinput(blk)
        byinput || (J = jacobian(blk, nT, varvals))
        # Compute direct Jacobians for each input-output pair
        for (i, vi) in enumerate(inputs(blk))
            # Only need variables that are reachable from sources
            if vi in vars
                byinput && (J = jacobian(blk, Val(i), nT, varvals))
                r0next = 0
                for (r, vo) in enumerate(outputs(blk))
                    # Input/output variable could be an array
                    Ni = length(varvals[vi])
                    No = outlength(blk, varvals, r)
                    # Make sure r0 is updated even if the iteration is skipped
                    r0 = r0next
                    r0next += No
                    excluded !== nothing && vo in excluded && continue
                    jo = get!(DMat, parts, vo)
                    mj = get(jo, vi, nothing)
                    hasj0 = mj !== nothing
                    breakloop = false
                    for ii in 1:Ni
                        breakloop && break
                        for rr in 1:No
                            # r0 is only used by SimpleBlock
                            j, isz = getjacmap(blk, J, i, ii, r, rr, r0, nT)
                            isz && continue
                            # Create the array mj only when nonzero map is encountered
                            if mj === nothing
                                if j isa MatMulMap
                                    mj = j # Need this for below
                                    jo[vi] = j
                                    breakloop = true
                                    break
                                elseif j isa WrappedMap
                                    mj = Matrix{typeof(j)}(undef, No, Ni)
                                else
                                    mj = Matrix{LinearMap{TF}}(undef, No, Ni)
                                    fill!(mj, zmap)
                                end
                                jo[vi] = mj
                            end
                            # j0 might be nonzero if multiple temporal inputs exist
                            if hasj0
                                j0 = mj[rr,ii]
                                if eltype(mj) <: WrappedMap
                                    mj[rr,ii] = LinearMap(Matrix(j0+j))
                                else
                                    mj[rr,ii] = j0 + j
                                end
                            else
                                mj[rr,ii] = j
                            end
                        end
                    end
                    # Do not bring zeros into total Jacobians
                    mj === nothing && continue
                    push!(vars, vo)
                    if !pushed
                        push!(blks, blk)
                        pushed = true
                    end
                end
            end
        end
        # Assemble maps for the total Jacobians
        # Jacobians from multiple temporal inputs are already combined
        for vi in unique(inputs(blk))
            if vi in vars
                for vo in outputs(blk)
                    excluded !== nothing && vo in excluded && continue
                    mj = get(parts[vo], vi, nothing)
                    mj === nothing && continue
                    # Handle the case when vi is a source of the DAG
                    unknown = get(totals, vi, nothing)
                    # If vi is not a source
                    if unknown === nothing
                        # Iterate over total Jacobians for each source
                        for d in values(totals)
                            maplast = get(d, vi, nothing)
                            if maplast !== nothing
                                # Maps from multiple temporal terms are already summed in mj
                                # maplast could be a row vector
                                if length(mj) == 1 &&
                                        !(maplast isa MatOfMap || maplast isa MatMulMap) &&
                                        mj isa Matrix
                                    mcomp = mj[1] * maplast
                                    # Combine possibly multiple channels
                                    map0 = get(d, vo, nothing)
                                    d[vo] = map0 === nothing ? mcomp : mcomp + map0
                                else
                                    mcomp = mapmatmul(mj, maplast)
                                    map0 = get(d, vo, nothing)
                                    if map0 === nothing
                                        d[vo] = mcomp
                                    elseif map0 isa MatMulMap || mcomp isa MatMulMap
                                        d[vo] = mcomp + map0
                                    else
                                        d[vo] = mcomp .+ map0
                                    end
                                end
                            end
                        end
                    # If vi is a source
                    else
                        # Combine possibly multiple channels
                        map0 = get(unknown, vo, nothing)
                        if length(mj) == 1 && mj isa Matrix
                            if map0 === nothing
                                unknown[vo] = mj[1]
                            elseif map0 isa Matrix
                                unknown[vo] = Ref(mj[1]) .+ map0
                            else
                                unknown[vo] = mj[1] + map0
                            end
                        else
                            if map0 === nothing
                                unknown[vo] = mj
                            elseif map0 isa MatMulMap || mj isa MatMulMap
                                unknown[vo] = mj + map0
                            else
                                unknown[vo] = mj .+ map0
                            end
                        end
                    end
                end
            end
        end
    end
    # Record the number of sources each target can be reached from
    ntarsrc = zeros(Int, length(targets))
    for d in values(totals)
        for (i, v) in enumerate(targets)
            if haskey(d, v)
                ntarsrc[i] += 1
            end
        end
    end
    any(ntarsrc.<2) && @warn "not all targets are reachable from at least two sources"
    return TotalJacobian(m, blks, vars, varvals, nT, sources, targets, ntarsrc, excluded, parts, totals)
end

inlength(j::TotalJacobian) = length(j.srcs)
tarlength(j::TotalJacobian) = length(j.tars)

show(io::IO, ::TotalJacobian{TF}) where TF = print(io, "TotalJacobian{$TF}")

function show(io::IO, ::MIME"text/plain", j::TotalJacobian{TF}) where TF
    nblk = length(j.blks)
    nvar = length(j.vars)
    print(io, "TotalJacobian{$TF} with $nblk block")
    nblk > 1 && print(io, 's')
    print(io, ", $nvar variable")
    nvar > 1 && print(io, 's')
    print(io, " and ", j.nT, " period")
    j.nT > 1 && print(io, 's')
    println(io, ":")
    print(io, "  sources: ")
    join(io, sort([j.srcs...]), ", ")
    print(io, "\n  targets: ")
    join(io, j.tars, ", ")
end

struct GEJacobian{TF<:AbstractFloat, HU<:Union{Matrix{TF},Nothing},
        FAC<:Union{<:Factorization,Nothing}}
    tjac::TotalJacobian{TF}
    exovars::Vector{Symbol}
    unknowns::Vector{Symbol}
    H_U::HU
    factor::FAC
    nTfull::Int
    Gs::Dict{Symbol,Dict{Symbol,JacMap{TF}}}
    Ms::Dict{Symbol,Dict{Symbol,Matrix{TF}}}
end

function _filljac!(out::Matrix, tjac::TotalJacobian, vars)
    nT = tjac.nT
    vals = tjac.varvals
    i0, j0 = 0, 0
    for v in vars
        tj = tjac.totals[v]
        N = length(vals[v])
        for t in tjac.tars
            M = length(vals[t])
            if haskey(tj, t)
                jac = tj[t]
                if jac isa LinearMap
                    _unsafe_mul!(view(out, i0+1:i0+M*nT, j0+1:j0+N*nT), jac, true)
                else
                    for n in 1:N
                        for m in 1:M
                            rr = i0+1+(m-1)*nT:i0+m*nT
                            rc = j0+1+(n-1)*nT:j0+n*nT
                            _unsafe_mul!(view(out, rr, rc), jac[m,n], true)
                        end
                    end
                end
            else
                fill!(view(out, i0+1:i0+M*nT, j0+1:j0+N*nT), zero(eltype(out)))
            end
            i0 += M * nT
        end
        i0 = 0
        j0 += N * nT
    end
    return out
end

_getvarlength(vars, vals::NamedTuple) = sum(v->length(vals[v]), vars)

function GEJacobian(tjac::TotalJacobian{TF}, exovars;
        keepH_U::Bool=false, keepfactor::Bool=false, nTfull::Int=tjac.nT) where TF
    exovars isa Symbol && (exovars = (exovars,))
    isempty(exovars) && throw(ArgumentError("exovars cannot be empty"))
    for var in exovars
        var in tjac.srcs || throw(ArgumentError("$var is not a source variable"))
    end
    exovars = collect(Symbol, exovars)
    unknowns = collect(setdiff(tjac.srcs, exovars))
    nT = tjac.nT
    vals = tjac.varvals
    Ntar = _getvarlength(tjac.tars, vals)
    NU = _getvarlength(unknowns, vals)
    NZ = _getvarlength(exovars, vals)
    H_U = Matrix{TF}(undef, Ntar*nT, NU*nT)
    _filljac!(H_U, tjac, unknowns)
    H_Z = Matrix{TF}(undef, Ntar*nT, NZ*nT)
    _filljac!(H_Z, tjac, exovars)
    keepH_U && (hu = copy(H_U))
    H_U = lu!(H_U)
    ldiv!(H_U, H_Z)

    G_U = H_Z
    G_U .*= -one(eltype(G_U))
    Gs = Dict{Symbol,Dict{Symbol,JacMap{TF}}}()
    Ms = Dict{Symbol,Dict{Symbol,Matrix{TF}}}()
    j0 = 0
    for z in exovars
        Gs[z] = Dict{Symbol,JacMap{TF}}()
        Ms[z] = Dict{Symbol,Matrix{TF}}()
        nz = length(tjac.varvals[z])
        i0 = 0
        for u in unknowns
            nu = length(tjac.varvals[u])
            if nu > 1 || nz > 1
                # Determine whether eltype of ms is WrappedMap by the first block
                # This is required for MatMulMap to work
                m1 = view(G_U, 1+i0*nT:(i0+1)*nT, 1+j0*nT:(j0+1)*nT)
                if length(m1) == 1 && nTfull > 1
                    ms = Matrix{LinearMap{TF}}(undef, nu, nz)
                    for jj in 1:nz
                        for ii in 1:nu
                            rr = 1+(i0+ii-1)*nT:(i0+ii)*nT
                            rc = 1+(j0+jj-1)*nT:(j0+jj)*nT
                            m = view(G_U, rr, rc)
                            ms[ii,jj] = UniformScalingMap(m[1], nTfull)
                        end
                    end
                else
                    ms = Matrix{WrappedMap{TF}}(undef, nu, nz)
                    for jj in 1:nz
                        for ii in 1:nu
                            rr = 1+(i0+ii-1)*nT:(i0+ii)*nT
                            rc = 1+(j0+jj-1)*nT:(j0+jj)*nT
                            m = view(G_U, rr, rc)
                            ms[ii,jj] = LinearMap(m)
                        end
                    end
                end
                Gs[z][u] = ms
            else
                m = view(G_U, 1+i0*nT:(i0+1)*nT, 1+j0*nT:(j0+1)*nT)
                if length(m) == 1 && nTfull > 1
                    Gs[z][u] = UniformScalingMap(m[1], nTfull)
                else
                    Gs[z][u] = LinearMap(m)
                end
            end
            i0 += nu
        end
        j0 += nz
    end
    return GEJacobian(tjac, exovars, unknowns,
        keepH_U ? hu : nothing, keepfactor ? H_U : nothing, nTfull, Gs, Ms)
end

show(io::IO, ::GEJacobian{TF}) where TF = print(io, "GEJacobian{$TF}")

function show(io::IO, ::MIME"text/plain", j::GEJacobian{TF}) where TF
    print(io, "GEJacobian{$TF} with ", j.tjac.nT, " period")
    j.tjac.nT > 1 && print(io, 's')
    println(io, ":")
    print(io, "  exogenous:  ")
    join(io, j.exovars, ", ")
    print(io, "\n  endogenous: ")
    join(io, j.unknowns, ", ")
    print(io, "\n  targets:    ")
    join(io, j.tjac.tars, ", ")
end

_resizemap(m::UniformScalingMap, nT) = UniformScalingMap(m.λ, nT)
_resizemap(m::ShiftMap, nT) = UniformScalingMap(m.S.v[1], nT)

function getG!(GJ::GEJacobian{TF}, exovar::Symbol, endovar::Symbol) where TF
    haskey(GJ.Gs, exovar) || throw(ArgumentError("$exovar is not an exogenous variable"))
    Gz = GJ.Gs[exovar]
    # G is readily available if endovar is a source
    haskey(Gz, endovar) && return Gz[endovar]
    # endovar does not have to be a source but must have been encountered by tjac
    endovar in GJ.tjac.vars ||
        throw(ArgumentError("$endovar is not an endogenous variable"))
    nexo = length(GJ.tjac.varvals[exovar])
    nendo = length(GJ.tjac.varvals[endovar])
    nT = GJ.nTfull
    zmap = LinearMap(UniformScaling(zero(TF)), nT)
    if nendo > 1 || nexo > 1
        M = Matrix{LinearMap{TF}}(undef, nendo, nexo)
        fill!(M, zmap)
    else
        M = zmap
    end
    # M_U combines all indirect effects while M_u is for a specific channel
    for (src, ms) in GJ.tjac.totals
        # Direct effect of exovar M_Z
        if src === exovar
            if haskey(ms, endovar)
                m = ms[endovar]
                if m isa LinearMap
                    size(m) == (1,1) && !(m isa MatMulMap) && (m = _resizemap(m, nT))
                    M += m
                else
                    if size(m[1,1]) == (1,1)
                        size(M) == size(m) || throw(DimensionMismatch())
                        for i in eachindex(M)
                            M[i] += _resizemap(m[i], nT)
                        end
                    else
                        M isa MatMulMap ? (M += m) : (M .+= m)
                    end
                end
            end
        # Indirect effect via unknowns M_U
        elseif src in GJ.unknowns
            if haskey(ms, endovar)
                M_u = ms[endovar]
                if M_u isa LinearMap
                    if M_u isa MatMulMap
                        M += mapmatmul(M_u, Gz[src])
                    else
                        size(M_u) == (1,1) && (M_u = _resizemap(M_u, nT))
                        if Gz[src] isa MatMulMap
                            M += mapmatmul(M_u, Gz[src])
                        else
                            M += M_u * Gz[src]
                        end
                    end
                else
                    size(M_u[1,1]) == (1,1) && (M_u = _resizemap.(M_u, nT))
                    M += mapmatmul(M_u, Gz[src])
                end
            end
        end
    end
    Gz[endovar] = M
    return M
end

function getM!(GJ::GEJacobian{TF}, exovar::Symbol, endovar::Symbol) where TF
    haskey(GJ.Ms, exovar) || throw(ArgumentError("$exovar is not an exogenous variable"))
    Mz = GJ.Ms[exovar]
    haskey(Mz, endovar) && return Mz[endovar]
    G = getG!(GJ, exovar, endovar)
    if G isa LinearMap
        M = Matrix(G)
    else # Matrix of LinearMaps
        nT = size(G[1], 1)
        m, n = size(G)
        M = Matrix{TF}(undef, m*nT, n*nT)
        for c in 1:n
            for r in 1:m
                _unsafe_mul!(view(M,1+(r-1)*nT:r*nT,1+(c-1)*nT:c*nT), G[r,c], true)
            end
        end
    end
    Mz[endovar] = M
    return M
end
