using Test
using SequenceJacobians

using Base: has_offset_axes
using CSV
using CodecZlib: GzipDecompressorStream
using DataFrames
using Distributions
using GSL
using JSON3
using LinearAlgebra
using LinearMaps
using LinearMaps: UniformScalingMap, WrappedMap, _unsafe_mul!
using LoopVectorization
using NLsolve
using Roots: Brent, Secant
using SequenceJacobians: jacbyinput

import SequenceJacobians: backwardsolver, forwardsolver

if VERSION >= v"1.7"
    using OpenBLAS32_jll
end

exampledata(name::Union{Symbol,String}) =
    CSV.read(pkgdir(SequenceJacobians)*"/data/$name.csv.gz", DataFrame)

function loadjson(name::Union{Symbol,String})
    stream = open(pkgdir(SequenceJacobians)*"/data/$name.json.gz") |> GzipDecompressorStream
    return copy(JSON3.read(read(stream, String)))
end

const tests = [
    "utils",
    "shift",
    "mapmatmul",
    "solvers",
    "lawofmotion",
    "blocks",
    "model",
    "jacobian",
    "irf",
    "twoasset",
    "Horvath",
    "estimation",
    "bayesian"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
