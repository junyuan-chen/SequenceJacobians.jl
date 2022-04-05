module SequenceJacobians

using FiniteDiff: finite_difference_gradient!, GradientCache
using ForwardDiff
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: BLAS, I, UniformScaling, LU, lu!, ldiv!, norm, dot, stride1
using LinearMaps
using Requires
using SplitApplyCombine: splitdimsview
using Statistics: mean
using Tullio: @tullio

import Base: ==, eltype, zero, show, convert
import Graphs: SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed,
    has_vertex, has_edge, inneighbors, outneighbors, neighborhood

# Reexport objects from Graphs
export SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed, has_vertex, has_edge,
    inneighbors, outneighbors

export supconverged,
       interpolate_y!,
       interpolate_coord!,
       setmin!,

       Shift,
       Lag,
       Lead,
       ShiftMap,

       AbstractRootSolver,
       NoRootSolver,
       AbstractVectorRootSolver,
       AbstractScalarRootSolver,
       AbstractSolverCache,
       solve!,

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
       nouts,
       outlength,
       SimpleBlock,
       block,
       steadystate!,
       jacobian,
       transition!,

       BlockOrVar,
       SequenceSpaceModel,
       model,
       srcs,
       sssrcs,
       dests,
       isblock,
       SteadyState,
       residuals!,
       criterion!,

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
       getlastpolicy,
       getdist,
       getlastdist,
       getdistendo,
       update!,
       backward!,
       backward_endo!,
       backward_steadystate!,
       backward_init!,
       backward_status,
       backward_converged,
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
       assetproc,

       TotalJacobian,
       GEJacobian,
       getG!,

       CombinedBlock,

       SolvedBlock,

       Transition,

       linirf,
       nlirf

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
include("transition.jl")
include("irf.jl")
include("examples/utils.jl")
include("examples/rbc.jl")
include("examples/KrusellSmith.jl")
include("examples/twoasset.jl")

function __init__()
    @require GSL = "92c85e6c-cbff-5e0c-80f7-495c94daaecd" begin
        if VERSION >= v"1.7" && Sys.isapple()
            if !(@isdefined OpenBLAS32_jll)
                @warn "Require using OpenBLAS32_jll for GSL"
            end
            @require OpenBLAS32_jll = "656ef2d0-ae68-5445-9ca0-591084a874a2" begin
                BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
            end
        end
        include("solvers/gsl.jl")
    end
    @require Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665" begin
        include("solvers/roots.jl")
    end
end

end # module
