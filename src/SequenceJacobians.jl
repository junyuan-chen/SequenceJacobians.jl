module SequenceJacobians

using Base: RefValue
using FFTW: rfft, irfft
using FiniteDiff: finite_difference_gradient!, GradientCache, default_relstep
using ForwardDiff
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: BLAS, I, UniformScaling, Diagonal, Factorization, LU, lu!,
    cholesky!, ldiv!, norm, dot, stride1
using LinearMaps
using MacroTools
using MacroTools: postwalk
using Requires
using SplitApplyCombine: splitdimsview
using Statistics: mean
using Tullio: @tullio

import Base: ==, eltype, zero, show, convert
import CommonSolve: solve!
import Graphs: SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed,
    has_vertex, has_edge, inneighbors, outneighbors, neighborhood
import StatsBase: autocov!, autocov

# Reexport
export solve!
export SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed, has_vertex, has_edge,
    inneighbors, outneighbors
export autocov!, autocov

export supconverged,
       interpolate_y!,
       interpolate_coord!,
       apply_coord!,
       setmin!,

       Shift,
       Lag,
       Lead,
       ShiftMap,

       NoRootSolver,
       isvectorrootsolver,
       isscalarrootsolver,
       isrootsolver,
       isrootsolvercache,
       rootsolvercache,

       VarSpec,
       var,
       lag,
       lead,
       name,
       shift,
       AbstractBlock,
       inputs,
       invars,
       ssinputs,
       outputs,
       hascache,
       outlength,
       SimpleBlock,
       block,
       steadystate!,
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

       BlockOrVar,
       SequenceSpaceModel,
       model,
       srcs,
       sssrcs,
       dests,
       isblock,
       SteadyState,
       getvarvals,
       getval,
       inlength,
       targets,
       tarlength,
       hastarget,
       residuals!,
       criterion!,

       TotalJacobian,
       GEJacobian,
       getG!,

       CombinedBlock,

       SolvedBlock,

       @simple,
       @implicit,

       Transition,

       linirf,
       nlirf,

       loglikelihood!

include("utils.jl")
include("shift.jl")
include("solvers/interface.jl")
include("block.jl")
include("hetagent.jl")
include("lawofmotion.jl")
include("hetblock.jl")
include("model.jl")
include("jacobian.jl")
include("combinedblock.jl")
include("solvedblock.jl")
include("macros.jl")
include("transition.jl")
include("irf.jl")
include("estimation.jl")
include("examples/utils.jl")
include("examples/rbc.jl")
include("examples/KrusellSmith.jl")
include("examples/twoasset.jl")

function __init__()
    @require GSL = "92c85e6c-cbff-5e0c-80f7-495c94daaecd" begin
        if VERSION >= v"1.7"
            if !(@isdefined OpenBLAS32_jll)
                @info "Use OpenBLAS32_jll for GSL"
            end
            @require OpenBLAS32_jll = "656ef2d0-ae68-5445-9ca0-591084a874a2" begin
                BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
            end
        end
        include("solvers/gsl.jl")
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
