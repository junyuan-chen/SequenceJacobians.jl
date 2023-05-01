abstract type AbstractLinearSolver end
abstract type DenseLinearSolver <: AbstractLinearSolver end
abstract type SparseLinearSolver <: AbstractLinearSolver end

mutable struct DenseLUSolver{F<:LU, B, X} <: DenseLinearSolver
    ws::LUWs
    fac::F
    b::B
    x::X
end

mutable struct UmfpackLUSolver{F, B, X} <: SparseLinearSolver
    fac::F
    b::B
    x::X
end

default_linsolvertype(sparse::Bool) = sparse ? UmfpackLUSolver : DenseLUSolver

function init(::Type{DenseLUSolver}, A::DenseArray, b, x)
    ws = LUWs(A)
    fac = LU(LAPACK.getrf!(ws, A)...)
    return DenseLUSolver(ws, fac, b, x)
end

solve!(s::DenseLUSolver) = ldiv!(s.x, s.fac, s.b)

function solve!(s::DenseLUSolver, A::DenseArray)
    s.fac = LU(LAPACK.getrf!(s.ws, A)...)
    return ldiv!(s.x, s.fac, s.b)
end

init(::Type{UmfpackLUSolver}, A::AbstractSparseMatrixCSC, b, x) =
    UmfpackLUSolver(lu(A), b, x)

solve!(s::UmfpackLUSolver) = ldiv!(s.x, s.fac, s.b)

function solve!(s::UmfpackLUSolver, A::AbstractSparseMatrixCSC)
    s.fac = lu!(s.fac, A)
    return ldiv!(s.x, s.fac, s.b)
end
