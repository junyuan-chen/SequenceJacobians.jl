struct SolvedBlock{B,J} <: AbstractBlock
    blk::B
    jac::J
    SolvedBlock(blk::SimpleBlock, jac::Dict{Int,<:Matrix}) =
        new{typeof(blk),typeof(jac)}(blk, jac)
    SolvedBlock(blk::HetBlock, jac::HetAgentJacCache) =
        new{typeof(blk),typeof(jac)}(blk, jac)
    SolvedBlock(blk::CombinedBlock{true}, jac::GEJacobian) =
        new{typeof(blk),typeof(jac)}(blk, jac)
    SolvedBlock(blk::CombinedBlock{false}, jac::TotalJacobian) =
        new{typeof(blk),typeof(jac)}(blk, jac)
end

block(b::SimpleBlock, Js::Dict{Int,<:Matrix}) = SolvedBlock(b, Js)
block(b::HetBlock, ca::HetAgentJacCache) = SolvedBlock(b, ca)
block(b::CombinedBlock{true}, gejac::GEJacobian) = SolvedBlock(b, gejac)
block(b::CombinedBlock{false}, tjac::TotalJacobian) = SolvedBlock(b, tjac)

inputs(b::SolvedBlock) = inputs(b.blk)
invars(b::SolvedBlock) = invars(b.blk)
ssinputs(b::SolvedBlock) = ssinputs(b.blk)
outputs(b::SolvedBlock) = outputs(b.blk)

hascache(b::SolvedBlock) = hascache(b.blk)
nouts(b::SolvedBlock) = nouts(b.blk)
outlength(b::SolvedBlock, r::Int) = outlength(b.blk, r)

steadystate!(b::SolvedBlock, varvals::AbstractDict) =
    error("SolvedBlock is not allowed for solving the steady state")

jacbyinput(::SolvedBlock{<:SimpleBlock}) = true
jacbyinput(::SolvedBlock) = false

jacobian(b::SolvedBlock{<:SimpleBlock}, i::Int, nT::Int,
    varvals::Dict{Symbol,<:ValType{TF}}) where TF = b.jac[i]

_getnT(ca::HetAgentJacCache) = ca.nT
_getnT(gejac::GEJacobian) = gejac.tjac.nT
_getnT(tjac::TotalJacobian) = tjac.nT

function jacobian(b::SolvedBlock, nT::Int, varvals::Dict{Symbol,<:ValType{TF}}) where TF
    jacnT = _getnT(b.jac)
    jacnT == nT || error("time horizons of solved Jacobians are $jacnT instead of $nT")
    return b.jac
end

getjacmap(b::SolvedBlock, J, i::Int, ii::Int, r::Int, rr::Int, nT::Int) =
    getjacmap(b.blk, J, i, ii, r, rr, nT)
