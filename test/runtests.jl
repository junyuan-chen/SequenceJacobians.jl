using Test
using SequenceJacobians

using Base: has_offset_axes
using GSL
using LinearAlgebra
using LoopVectorization
using NLsolve
using Roots: Brent, Secant
using SequenceJacobians: jacbyinput

import SequenceJacobians: backwardsolver, forwardsolver

if VERSION >= v"1.7"
    using OpenBLAS32_jll
end

const tests = [
    "utils",
    "shift",
    "solvers",
    "lawofmotion",
    "blocks",
    "model",
    "jacobian",
    "twoasset",
    "estimation"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
