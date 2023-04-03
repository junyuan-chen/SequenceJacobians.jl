using Test
using SequenceJacobians

using AdvancedMH
using Base: has_offset_axes
using BlockArrays
using CSV
using CodecZlib: GzipDecompressorStream
using DataFrames: DataFrame, ncol, nrow
using Distributions
using DynamicHMC
using GSL
using JSON3
using LinearAlgebra
using LogDensityProblems: logdensity, logdensity_and_gradient
using LoopVectorization
using MCMCChains
using NLopt
using NLsolve
using NonlinearSystems
using Random
using Roots: Brent, Secant
using SequenceJacobians: ArrayToArgs, ar1impulse!, _solve!, _reshape
using StructArrays
using TransformVariables: as, asℝ₊, as𝕀

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

struct ImpulseResidual{U<:ImpulseUpdate, V<:AbstractVector}
    u::U
    tars::V
end

function (r::ImpulseResidual)(resids::AbstractVector, θ)
    r.u(θ)
    resids .= _reshape(r.u.vals, length(r.u.vals)) .- r.tars
    return resids
end

const tests = [
    "utils",
    "shift",
    "solvers",
    "lawofmotion",
    "blocks",
    "model",
    "jacobian",
    "shock",
    "twoasset",
    "Horvath",
    "estimation",
    "bayesian",
    "SmetsWouters"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
