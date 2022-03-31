module SequenceJacobians

using FiniteDiff: finite_difference_gradient!, GradientCache
using ForwardDiff
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: BLAS, I, UniformScaling, LU, lu!, ldiv!, norm, dot, stride1
using LinearMaps
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

       Transition,
       solve!,

       linirf,
       nlirf

include("utils.jl")
include("shift.jl")
include("block.jl")
include("hetagent.jl")
include("lawofmotion.jl")
include("hetblock.jl")
include("model.jl")
include("jacobian.jl")
include("transition.jl")
include("irf.jl")
include("example/utils.jl")
include("example/rbc.jl")
include("example/KrusellSmith.jl")

end # module
