const VarJacobian{TF} = Union{Shift{TF}, AbstractMatrix{TF}}

abstract type AbstractJacobianMap{TF} end

struct ShiftMap{TF} <: AbstractJacobianMap{TF}
    ins::Vector{Union{Matrix{TF}, Bool}}
    inshifts::Vector{Vector{Union{CompositeShift{TF}, Bool}}}
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
const ShiftOrBool{TF} = Union{CompositeShift{TF}, Bool}
const SMapOrNo{TF} = Union{ShiftMap{TF}, Nothing}

jacmap(S::Shift{TF}) where TF =
    ShiftMap(MatOrBool{TF}[true], [ShiftOrBool{TF}[true]], [Shift{TF}[S]],
        CompositeShift{TF}[S * true])
jacmap(M::AbstractMatrix{TF}) where TF =
    MatrixMap(SMapOrNo{TF}[nothing], MatOrBool{TF}[true], MatOrSub{TF}[M], copy(M))

copy(M::MatrixMap) = MatrixMap(copy(M.inmaps), copy(M.ins), copy(M.maps), copy(M.out))

function (*)(S::Shift{TF}, Slast::ShiftMap{TF}) where TF
    ins = copy(Slast.ins)
    inshifts = [ShiftOrBool{TF}[s] for s in Slast.outs]
    maps = [Shift{TF}[S] for _ in eachindex(Slast.outs)]
    outs = CompositeShift{TF}[S * s for s in Slast.outs]
    return ShiftMap(ins, inshifts, maps, outs)
end

(*)(S::Shift{TF}, Mlast::MatrixMap{TF}) where TF =
    ShiftMap(MatOrBool{TF}[Mlast.out], [ShiftOrBool{TF}[true]], [Shift{TF}[S]],
        CompositeShift{TF}[S * true])

# S is from a source variable
function muladd!(Smap::ShiftMap{TF}, S::Shift{TF}) where TF
    k = findfirst(x->x===true, Smap.ins)
    if k === nothing
        push!(Smap.ins, true)
        push!(Smap.inshifts, ShiftOrBool{TF}[true])
        push!(Smap.maps, Shift{TF}[S])
        push!(Smap.outs, S * true)
    else
        push!(Smap.inshifts[k], true)
        push!(Smap.maps[k], S)
        mul!(Smap.outs[k], S, true, true, true)
    end
    return Smap
end

function muladd!(Smap::ShiftMap{TF}, S::Shift{TF}, Slast::ShiftMap{TF}) where TF
    for i in eachindex(Slast.ins)
        k = findfirst(x->x===Slast.ins[i], Smap.ins)
        if k === nothing
            push!(Smap.ins, Slast.ins[i])
            push!(Smap.inshifts, ShiftOrBool{TF}[Slast.outs[i]])
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

function muladd!(Smap::ShiftMap{TF}, S::Shift{TF}, Mlast::MatrixMap{TF}) where TF
    k = findfirst(x->x===Mlast.out, Smap.ins)
    if k === nothing
        push!(Smap.ins, Mlast.out)
        push!(Smap.inshifts, ShiftOrBool{TF}[true])
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
        mul!(C, S.outs[i], S.ins[i], s, true)
    end
    return C
end

# All matrix ins in ShiftMap should have the same shape
function (S::ShiftMap{TF})(N::Int) where TF
    k = findfirst(x->x!==true, S.ins)
    if k === nothing
        return mul!(zeros(TF, N, N), S, true, true)
    else # Ignore N
        return mul!(similar(S.ins[k]), S, true, false)
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

function (*)(M::AbstractMatrix{TF}, Slast::ShiftMap{TF}) where TF
    m = Slast(size(M, 2))
    return MatrixMap(SMapOrNo{TF}[Slast], MatOrBool{TF}[m], MatOrSub{TF}[M], M*m)
end

(*)(M::AbstractMatrix{TF}, Mlast::MatrixMap{TF}) where TF =
    MatrixMap(SMapOrNo{TF}[nothing], MatOrBool{TF}[Mlast.out], MatOrSub{TF}[M], M*Mlast.out)

function muladd!(Mmap::MatrixMap{TF}, M::MatOrSub{TF}) where TF
    push!(Mmap.inmaps, nothing)
    push!(Mmap.ins, true)
    push!(Mmap.maps, M)
    mul!(Mmap.out, M, true, true, true)
    return Mmap
end

function muladd!(Mmap::MatrixMap{TF}, M::MatOrSub{TF}, Slast::ShiftMap{TF}) where TF
    push!(Mmap.inmaps, Slast)
    m = Slast(size(M, 2))
    push!(Mmap.ins, m)
    push!(Mmap.maps, M)
    mul!(Mmap.out, M, m, true, true)
    return Mmap
end

function muladd!(Mmap::MatrixMap{TF}, M::MatOrSub{TF}, Mlast::MatrixMap{TF}) where TF
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

mul!(C::AbstractVecOrMat, M::MatrixMap, s::Number, β::Number=false) =
    mul!(C, M.out, s, true, β)
