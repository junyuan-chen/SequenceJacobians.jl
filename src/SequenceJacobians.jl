module SequenceJacobians

using AutoregressiveModels: ARMAProcess
using Base: RefValue, ReshapedArray
using BlockArrays: BlockMatrix, PseudoBlockMatrix, PseudoBlockVector, Block,
    BlockedUnitRange, mortar, blocksize, blocksizes, _BlockedUnitRange,
    MemoryLayout, _copyto!
using Distributions: Distribution, logpdf
using FFTW: Plan, plan_rfft, plan_irfft, rfft, irfft
using FastLapackInterface: LUWs
using FillArrays: Fill, Zeros
using FiniteDiff: JacobianCache, finite_difference_jacobian!, GradientCache,
    finite_difference_gradient!, HessianCache, finite_difference_hessian!, default_relstep
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: BLAS, LAPACK, I, UniformScaling, Diagonal, LU, lu!, rmul!,
    Hermitian, cholesky!, ldiv!, inv!, norm, dot, stride1, diag, diagind
using LogDensityProblems: LogDensityOrder
using MacroTools
using MacroTools: postwalk
using Printf
using Requires
using SplitApplyCombine: splitdimsview
using Statistics: mean
using Tables
using TransformVariables: AbstractTransform, transform_logdensity
using TransformedLogDensities: TransformedLogDensity
using Tullio: @tullio

import AutoregressiveModels: simulate!, simulate, impulse!, impulse
import Base: ==, eltype, show, Matrix, parent, getindex, copy, convert, iterate, length,
    ndims, has_offset_axes, zero, iszero, +, *
import CommonSolve: solve!
import Graphs: SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed,
    has_vertex, has_edge, inneighbors, outneighbors, neighborhood
import LinearAlgebra: isdiag, mul!
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
import StatsAPI: vcov, stderror
import StatsBase: mode
import TransformVariables: transform

# Reexport
export ARMAProcess, simulate!, simulate, impulse!, impulse
export solve!
export SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed, has_vertex, has_edge,
    inneighbors, outneighbors
export dimension
export vcov, stderror
export mode
export transform

export supconverged,
       interpolate_y!,
       interpolate_coord!,
       apply_coord!,
       setmin!,
       acceptance_rate,

       Shift,
       CompositeShift,

       AbstractJacobianMap,
       ShiftMap,
       MatrixMap,
       jacmap,

       NoRootSolver,
       isvectorrootsolver,
       isscalarrootsolver,
       isrootsolver,
       isrootsolvercache,
       rootsolvercache,

       VarSpec,
       varspec,
       lag,
       lead,
       name,
       shift,
       AbstractBlock,
       inputs,
       invars,
       ssinputs,
       outputs,
       outlength,
       SimpleBlock,
       block,
       steadystate!,
       AbstractBlockJacobian,
       SimpleBlockJacobian,
       jacobian,
       transition!,

       AbstractHetAgent,
       HetAgentStyle,
       TimeDiscrete,
       endostates,
       getendo,
       endopolicies,
       exogstates,
       getexog,
       statevars,
       valuevars,
       getvalue,
       getexpectedvalue,
       policies,
       getpolicy,
       backwardtargets,
       getbackwardtarget,
       getlastbackwardtarget,
       getdist,
       getlastdist,
       getdistendo,
       update!,
       backwardsolver,
       backward!,
       backward_endo!,
       backward_steadystate!,
       backward_init!,
       backward_status,
       backward_converged,
       forwardsolver,
       forward!,
       forward_steadystate!,
       forward_init!,
       forward_status,
       forward_converged,
       aggregate,

       AbstractLawOfMotion,
       grid,
       DiscreteTimeLawOfMotion,
       ExogProc,
       rouwenhorstexp,
       EndoProc,
       assetgrid,
       assetproc,

       HetBlock,
       HetAgentJacCache,
       HetBlockJacobian,

       BlockOrVar,
       SequenceSpaceModel,
       model,
       srcs,
       vsrcs,
       sssrcs,
       vsssrcs,
       dests,
       vdests,
       isblock,
       SteadyState,
       varvalstype,
       blkstype,
       scalarinputs,
       arrayinputs,
       scalartargets,
       arraytargets,
       inlength,
       targets,
       tarlength,
       hastarget,
       residuals!,
       criterion!,

       TotalJacobian,
       GEJacobian,
       GEJacobianUpdatePlan,
       plan,
       GMaps,

       CombinedBlock,
       CombinedBlockJacobian,

       @simple,
       @implicit,

       AbstractAllCovCache,
       FFTWAllCovCache,
       allcov!,
       allcov,
       allcor!,
       allcor,
       correlogram!,
       correlogram,
       loglikelihood!,

       ShockProcess,
       shockse,
       ar1shock,
       arma11shock,
       aswidetable,
       aslongtable,

       BayesianModel,
       TransformedBayesianModel,
       BayesOrTrans,
       bayesian,
       nshock,
       nshockpara,
       nstrucpara,
       logprior,
       logposterior!,
       logposterior_and_gradient!,
       logdensity_hessian!,
       vcov!,

       MinimumDistance

include("utils.jl")
include("shift.jl")
include("jacmap.jl")
include("solvers/interface.jl")
include("block.jl")
include("hetagent.jl")
include("lawofmotion.jl")
include("hetblock.jl")
include("model.jl")
include("jacobian.jl")
include("combinedblock.jl")
include("macros.jl")
include("allcov.jl")
include("shock.jl")
include("bayesian.jl")
include("mindist.jl")
include("examples/utils.jl")
include("examples/rbc.jl")
include("examples/KrusellSmith.jl")
include("examples/twoasset.jl")
include("examples/Horvath.jl")
include("examples/SmetsWouters.jl")

function __init__()
    @require GSL = "92c85e6c-cbff-5e0c-80f7-495c94daaecd" begin
        if VERSION >= v"1.7"
            if !(@isdefined OpenBLAS32_jll)
                @info "Need to use OpenBLAS32_jll for GSL"
            end
            @require OpenBLAS32_jll = "656ef2d0-ae68-5445-9ca0-591084a874a2" begin
                BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
            end
        end
        include("solvers/gsl.jl")
    end
    @require NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd" begin
        include("solvers/nlopt.jl")
    end
    @require NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56" begin
        include("solvers/nlsolve.jl")
        include("solvers/anderson.jl")
    end
    @require Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665" begin
        include("solvers/roots.jl")
    end
end

end # module
