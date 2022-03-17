using Test
using SequenceJacobians

using Base: has_offset_axes
using LinearAlgebra
using MINPACK: fsolve
using SequenceJacobians: ValType

const tests = [
    "utils",
    "shift",
    "blocks",
    "model",
    "jacobian"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
