module SequenceJacobians

using ForwardDiff
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: I, UniformScaling, LU, lu!, ldiv!, norm
using LinearMaps

import Base: ==, eltype, zero, show, convert
import Graphs: SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed,
    has_vertex, has_edge, inneighbors, outneighbors, neighborhood

# Reexport objects from Graphs
export SimpleDiGraph, edgetype, nv, ne, vertices, edges, is_directed, has_vertex, has_edge,
    inneighbors, outneighbors

export Shift,
       Lag,
       Lead,
       ShiftMap,

       VarInput,
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

       TotalJacobian,
       GEJacobian,
       getG!,

       Transition,
       update!,
       solve!,

       linirf,
       nlirf

include("shift.jl")
include("blocks.jl")
include("model.jl")
include("jacobian.jl")
include("transition.jl")
include("irf.jl")

end # module
