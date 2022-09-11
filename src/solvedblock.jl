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
block(b::CombinedBlock, GJ::GEJacobian) = SolvedBlock(b, GJ)
block(b::CombinedBlock{0}, J::TotalJacobian) = SolvedBlock(b, J)

invars(b::SolvedBlock) = invars(b.blk)
ssinputs(b::SolvedBlock) = ssinputs(b.blk)

outlength(b::SolvedBlock, varvals::NamedTuple) = outlength(b.blk, varvals)
outlength(b::SolvedBlock, varvals::NamedTuple, r::Int) = outlength(b.blk, varvals, r)

steadystate!(b::SolvedBlock, varvals::NamedTuple) =
    error("SolvedBlock is not allowed for solving the steady state")

jacbyinput(::SolvedBlock{<:SimpleBlock}) = true
jacbyinput(::SolvedBlock) = false

jacobian(b::SolvedBlock{<:SimpleBlock}, ::Val{i}, nT::Int, varvals::NamedTuple) where i =
    b.jac[i]

_getnT(ca::HetAgentJacCache) = ca.nT
_getnT(GJ::GEJacobian) = GJ.tjac.nT
_getnT(J::TotalJacobian) = J.nT

function jacobian(b::SolvedBlock, nT::Int, varvals::NamedTuple) where TF
    jacnT = _getnT(b.jac)
    jacnT == nT || error("time horizons of solved Jacobians are $jacnT instead of $nT")
    return b.jac
end

getjacmap(b::SolvedBlock, J, i::Int, ii::Int, r::Int, rr::Int, r0::Int, nT::Int) =
    getjacmap(b.blk, J, i, ii, r, rr, r0, nT)

show(io::IO, b::SolvedBlock) = print(io, "SolvedBlock($(b.blk))")

function show(io::IO, ::MIME"text/plain", b::SolvedBlock)
    println(io, "SolvedBlock($(b.blk)):")
    _showinouts(io, b)
end
