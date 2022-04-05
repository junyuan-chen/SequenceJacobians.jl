using Test
using SequenceJacobians

using Base: has_offset_axes
using GSL
using LinearAlgebra
using LoopVectorization
using Roots: Brent
using SequenceJacobians: ValType, jacbyinput

if VERSION >= v"1.7" && Sys.isapple()
    using OpenBLAS32_jll
end

const tests = [
    "utils",
    "shift",
    "solvers",
    "lawofmotion",
    "blocks",
    "model",
    "jacobian"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
