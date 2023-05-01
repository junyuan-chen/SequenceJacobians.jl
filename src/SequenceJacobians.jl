module SequenceJacobians

using AutoregressiveModels: ARMAProcess
using Base: RefValue, ReshapedArray, Callable, Fix1, Fix2, @propagate_inbounds
using BlockArrays: BlockMatrix, PseudoBlockMatrix, PseudoBlockVector, Block,
    BlockedUnitRange, mortar, blocksize, blocksizes, _BlockedUnitRange,
    MemoryLayout, _copyto!
using CommonSolve: solve
using ComponentArrays: Axis, ComponentArray, ComponentVector
using Distributions: Distribution, logpdf
using FFTW: Plan, plan_rfft, plan_irfft, rfft, irfft
using FastLapackInterface: LUWs
using FillArrays: Fill, Zeros
using FiniteDiff: JacobianCache, finite_difference_jacobian!, GradientCache,
    finite_difference_gradient!, HessianCache, finite_difference_hessian!, default_relstep
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: BLAS, LAPACK, I, UniformScaling, Diagonal, LU, lu!, lu, rmul!,
    Hermitian, cholesky!, ldiv!, inv!, norm, dot, stride1, diag, diagind
using LogDensityProblems: LogDensityOrder
using MacroTools
using MacroTools: postwalk
using Printf
using Requires
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC, spdiagm, spzeros,
    getcolptr, getnzval, nzrange
using SplitApplyCombine: splitdimsview
using StaticArraysCore: SVector
using Statistics: mean
using StatsBase: _denserank!
using StructArrays: StructArray
using Tables
using TransformVariables: AbstractTransform, transform_logdensity
using TransformedLogDensities: TransformedLogDensity
using Tullio: @tullio

import AutoregressiveModels: simulate!, simulate, impulse!, impulse
import Base: ==, eltype, show, Matrix, parent, getindex, copy, convert, iterate, length,
    ndims, has_offset_axes, zero, iszero, +, *
import CommonSolve: init, solve!
import Graphs: SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed,
    has_vertex, has_edge, inneighbors, outneighbors, neighborhood
import LinearAlgebra: isdiag, mul!
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
import SparseArrays: sparse
import StatsAPI: vcov, stderror
import StatsBase: mode
import TransformVariables: transform

# Reexport
export ARMAProcess, simulate!, simulate, impulse!, impulse
export solve!, solve
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
       rowblocks,
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
       backwardsolvercache,
       forwardsolvercache,
       Roots_Default,
       NLsolve_newton,
       NLsolve_trust_region,
       NLsolve_anderson,
       NLsolve_broyden,
       NLsolve_Solver,
       BroydenCache,
       NLsolve_Cache,

       AbstractLinearSolver,
       DenseLinearSolver,
       SparseLinearSolver,
       DenseLUSolver,
       UmfpackLUSolver,

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
       backward_exog!,
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

       ImpulseUpdate,

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
       vcov!

include("utils.jl")
include("shift.jl")
include("jacmap.jl")
include("solverinterface.jl")
include("linearsolver.jl")
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
include("impulseupdate.jl")
include("bayesian.jl")
include("examples/utils.jl")
include("examples/rbc.jl")
include("examples/KrusellSmith.jl")
include("examples/twoasset.jl")
include("examples/Horvath.jl")
include("examples/SmetsWouters.jl")

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd" begin
            include("../ext/SeqJacNLoptExt.jl")
        end
        @require NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56" begin
            export splitdimsview
            include("../ext/SeqJacNLsolveExt.jl")
        end
        @require NonlinearSystems = "deb0877a-d74a-4aeb-b6ac-c17f6fb4122e" begin
            include("../ext/SeqJacNonlinearSystemsExt.jl")
        end
        @require Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665" begin
            include("../ext/SeqJacRootsExt.jl")
        end
    end
end

end # module
