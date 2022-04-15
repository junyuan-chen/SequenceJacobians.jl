struct SolvedBlock{B,J,ins,outs} <: AbstractBlock{ins,outs}
    blk::B
    jac::J
    SolvedBlock(blk::SimpleBlock, jac::Dict{Int,<:Matrix}) =
        new{typeof(blk),typeof(jac),inputs(blk),outputs(blk)}(blk, jac)
    SolvedBlock(blk::HetBlock, jac::HetAgentJacCache) =
        new{typeof(blk),typeof(jac),inputs(blk),outputs(blk)}(blk, jac)
    SolvedBlock(blk::CombinedBlock, jac::GEJacobian) =
        new{typeof(blk),typeof(jac),inputs(blk),outputs(blk)}(blk, jac)
    SolvedBlock(blk::CombinedBlock{0}, jac::TotalJacobian) =
        new{typeof(blk),typeof(jac),inputs(blk),outputs(blk)}(blk, jac)
end

block(b::SimpleBlock, Js::Dict{Int,<:Matrix}) = SolvedBlock(b, Js)
block(b::HetBlock, ca::HetAgentJacCache) = SolvedBlock(b, ca)
block(b::CombinedBlock, gejac::GEJacobian) = SolvedBlock(b, gejac)
block(b::CombinedBlock{0}, tjac::TotalJacobian) = SolvedBlock(b, tjac)

invars(b::SolvedBlock) = invars(b.blk)
ssinputs(b::SolvedBlock) = ssinputs(b.blk)

hascache(b::SolvedBlock) = hascache(b.blk)
outlength(b::SolvedBlock) = outlength(b.blk)
outlength(b::SolvedBlock, r::Int) = outlength(b.blk, r)

steadystate!(b::SolvedBlock, varvals::NamedTuple) =
    error("SolvedBlock is not allowed for solving the steady state")

jacbyinput(::SolvedBlock{<:SimpleBlock}) = true
jacbyinput(::SolvedBlock) = false

jacobian(b::SolvedBlock{<:SimpleBlock}, ::Val{i}, nT::Int, varvals::NamedTuple) where i =
    b.jac[i]

_getnT(ca::HetAgentJacCache) = ca.nT
_getnT(gejac::GEJacobian) = gejac.tjac.nT
_getnT(tjac::TotalJacobian) = tjac.nT

function jacobian(b::SolvedBlock, nT::Int, varvals::NamedTuple) where TF
    jacnT = _getnT(b.jac)
    jacnT == nT || error("time horizons of solved Jacobians are $jacnT instead of $nT")
    return b.jac
end

getjacmap(b::SolvedBlock, J, i::Int, ii::Int, r::Int, rr::Int, nT::Int) =
    getjacmap(b.blk, J, i, ii, r, rr, nT)

show(io::IO, b::SolvedBlock) = print(io, "SolvedBlock($(b.blk))")

function show(io::IO, ::MIME"text/plain", b::SolvedBlock)
    println(io, "SolvedBlock($(b.blk)):")
    _showinouts(io, b)
end
