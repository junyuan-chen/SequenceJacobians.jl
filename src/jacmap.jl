abstract type AbstractJacobianMap{TF} end

struct ShiftMap{TF} <: AbstractJacobianMap{TF}
    ins::Vector{Union{Matrix{TF}, Bool}}
    inshifts::Vector{Vector{Union{CompositeShift, Bool}}}
    maps::Vector{Vector{Shift{TF}}}
    outs::Vector{CompositeShift{TF}}
end

struct MatrixMap{TF} <: AbstractJacobianMap{TF}
    inmaps::Vector{Union{ShiftMap{TF}, Nothing}}
    ins::Vector{Union{Matrix{TF}, Bool}}
    maps::Vector{MatOrSub{TF}}
    out::Matrix{TF}
end

const MatOrBool{TF} = Union{Matrix{TF}, Bool}
# Incomplete parameters for CompositeShift would cause StackOverflowError on Julia v1.6
const ShiftOrBool = Union{CompositeShift, Bool}
const SMapOrNo{TF} = Union{ShiftMap{TF}, Nothing}

# inmat is used in only two scenarios:
# 1) impulse responses is computed only for specific shock paths specified with dZs
# 2) MatrixMaps from the same block share the same ins for previous ShiftMaps

jacmap(S::Shift{TF}, inmat=true) where TF =
    ShiftMap(MatOrBool{TF}[inmat], [ShiftOrBool[true]], [Shift{TF}[S]],
        CompositeShift{TF}[S * true])

jacmap(M::AbstractMatrix{TF}, inmat=true) where TF =
    MatrixMap(SMapOrNo{TF}[nothing], MatOrBool{TF}[inmat], MatOrSub{TF}[M], M * inmat)

function jacmap(S::Shift{TF}, Slast::ShiftMap{TF}, inmat=nothing) where TF
    ins = copy(Slast.ins)
    inshifts = [ShiftOrBool[s] for s in Slast.outs]
    maps = [Shift{TF}[S] for _ in eachindex(Slast.outs)]
    outs = CompositeShift{TF}[S * s for s in Slast.outs]
    return ShiftMap(ins, inshifts, maps, outs)
end

jacmap(S::Shift{TF}, Mlast::MatrixMap{TF}, inmat=nothing) where TF =
    ShiftMap(MatOrBool{TF}[Mlast.out], [ShiftOrBool[true]], [Shift{TF}[S]],
        CompositeShift{TF}[S * true])

# S is from a source variable
function muladd!(Smap::ShiftMap{TF}, S::Shift{TF}, inmat::Union{Matrix{TF},Bool}) where TF
    k = findfirst(x->x===inmat, Smap.ins)
    if k === nothing
        push!(Smap.ins, inmat)
        push!(Smap.inshifts, ShiftOrBool[true])
        push!(Smap.maps, Shift{TF}[S])
        push!(Smap.outs, S * true)
    else
        push!(Smap.inshifts[k], true)
        push!(Smap.maps[k], S)
        mul!(Smap.outs[k], S, true, true, true)
    end
    return Smap
end

function muladd!(Smap::ShiftMap{TF}, S::Shift{TF}, Slast::ShiftMap{TF}, inmat=nothing) where TF
    for i in eachindex(Slast.ins)
        k = findfirst(x->x===Slast.ins[i], Smap.ins)
        if k === nothing
            push!(Smap.ins, Slast.ins[i])
            push!(Smap.inshifts, ShiftOrBool[Slast.outs[i]])
            push!(Smap.maps, Shift{TF}[S])
            push!(Smap.outs, S * Slast.outs[i])
        else
            push!(Smap.inshifts[k], Slast.outs[i])
            push!(Smap.maps[k], S)
            mul!(Smap.outs[k], S, Slast.outs[i], true, true)
        end
    end
    return Smap
end

function muladd!(Smap::ShiftMap{TF}, S::Shift{TF}, Mlast::MatrixMap{TF}, inmat=nothing) where TF
    k = findfirst(x->x===Mlast.out, Smap.ins)
    if k === nothing
        push!(Smap.ins, Mlast.out)
        push!(Smap.inshifts, ShiftOrBool[true])
        push!(Smap.maps, Shift{TF}[S])
        push!(Smap.outs, S * true)
    else
        push!(Smap.inshifts[k], true)
        push!(Smap.maps[k], S)
        mul!(Smap.outs[k], S, true, true, true)
    end
    return Smap
end

function mul!(C::AbstractVecOrMat, S::ShiftMap, s::Number, β::Number=false)
    iszero(β) ? fill!(C, zero(eltype(C))) : isone(β) ? C : rmul!(C, β)
    for i in eachindex(S.outs)
        ini = S.ins[i]
        if ini isa Bool
            mul!(C, S.outs[i], ini, s, true)
        else
            # C is allowed to be smaller than S in the time dimension
            # This is useful for trimming the values at the end
            if S.outs[1].size == (1, 1)
                mul!(C, S.outs[i], view(ini, 1:size(C,1), 1:size(C,2)), s, true)
            else
                nT = Int(size(C,1)/S.outs[1].size[1])
                if nT * size(C,2) < size(ini, 1) # ! Does it work?
                    inib = _block1(ini, Int(size(ini,1)/S.outs[1].size[2]))
                    rb = blocksize(inib,1)
                    blks = [view(view(inib, ib, :), 1:nT, 1:size(C,2)) for ib in 1:rb]
                    mul!(C, S.outs[i], mortar(_reshape(blks, rb, 1)), s, true)
                else
                    mul!(C, S.outs[i], ini, s, true)
                end
            end
        end
    end
    return C
end

# All matrix ins in ShiftMap should have the same shape
function (S::ShiftMap{TF})(nT::Int) where TF
    k = findfirst(x->x!==true, S.ins)
    if k === nothing
        return mul!(zeros(TF, nT.*S.outs[1].size), S, true, true)
    else
        out = similar(S.ins[k], (nT*S.outs[1].size[1], size(S.ins[k],2)))
        return mul!(out, S, true, false)
    end
end

function _updateout!(Smap::ShiftMap)
    @inbounds for i in eachindex(Smap.outs)
        mul!(Smap.outs[i], Smap.maps[i][1], Smap.inshifts[i][1], true, false)
        for j in 2:length(Smap.maps[i])
            mul!(Smap.outs[i], Smap.maps[i][j], Smap.inshifts[i][j], true, true)
        end
    end
    return Smap
end

jacmap(M::AbstractMatrix{TF}, Slast::ShiftMap{TF}, inmat) where TF =
    MatrixMap(SMapOrNo{TF}[Slast], MatOrBool{TF}[inmat], MatOrSub{TF}[M], M*inmat)

jacmap(M::AbstractMatrix{TF}, Mlast::MatrixMap{TF}, inmat=nothing) where TF =
    MatrixMap(SMapOrNo{TF}[nothing], MatOrBool{TF}[Mlast.out], MatOrSub{TF}[M], M*Mlast.out)

# M is from a source variable
function muladd!(Mmap::MatrixMap{TF}, M::MatOrSub{TF}, inmat::Union{Matrix{TF},Bool}) where TF
    push!(Mmap.inmaps, nothing)
    push!(Mmap.ins, inmat)
    push!(Mmap.maps, M)
    mul!(Mmap.out, M, inmat, true, true)
    return Mmap
end

function muladd!(Mmap::MatrixMap{TF}, M::MatOrSub{TF}, Slast::ShiftMap{TF}, inmat) where TF
    push!(Mmap.inmaps, Slast)
    push!(Mmap.ins, inmat)
    push!(Mmap.maps, M)
    mul!(Mmap.out, M, inmat, true, true)
    return Mmap
end

function muladd!(Mmap::MatrixMap{TF}, M::MatOrSub{TF}, Mlast::MatrixMap{TF}, inmat=nothing) where TF
    push!(Mmap.inmaps, nothing)
    push!(Mmap.ins, Mlast.out)
    push!(Mmap.maps, M)
    mul!(Mmap.out, M, Mlast.out, true, true)
    return Mmap
end

function _updateout!(Mmap::MatrixMap)
    @inbounds mul!(Mmap.out, Mmap.maps[1], Mmap.ins[1], true, false)
    @inbounds for i in 2:lastindex(Mmap.maps)
        mul!(Mmap.out, Mmap.maps[i], Mmap.ins[i], true, true)
    end
    return Mmap
end

function _updateins!(Mmap::MatrixMap, iins::Union{Vector{Int},Nothing}=nothing)
    ids = iins === nothing ? eachindex(Mmap.inmaps) : iins
    for i in ids
        Smap = Mmap.inmaps[i]
        Smap === nothing && continue
        mul!(Mmap.ins[i], Smap, true)
    end
    return nothing
end

# C is allowed to be smaller than M
#! To do: What if out is for a BlockArray?
mul!(C::AbstractVecOrMat, M::MatrixMap, s::Number, β::Number=false) =
    mul!(C, view(M.out, 1:size(C,1), 1:size(C,2)) , s, true, β)

show(io::IO, S::ShiftMap{TF}) where TF =
    print(io, "ShiftMap{", TF, "}(", length(S.outs), ')')

function show(io::IO, ::MIME"text/plain", S::ShiftMap{TF}) where TF
    N = length(S.outs)
    print(io, "ShiftMap{", TF, "} with ", N, " component")
    print(io, N > 1 ? "s:" : ":")
    for out in S.outs
        print(io, "\n  ", out)
    end
end

show(io::IO, M::MatrixMap{TF}) where TF =
    print(io, "MatrixMap{", TF, "}(", length(M.maps), ')')

function show(io::IO, ::MIME"text/plain", M::MatrixMap{TF}) where TF
    N = length(M.maps)
    print(io, "MatrixMap{", TF, "} combined from ", N, " component")
    println(io, N > 1 ? "s:" : ":")
    print(IOContext(io, :compact=>true), "  ", M.out)
end
