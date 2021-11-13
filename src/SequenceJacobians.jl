module SequenceJacobians

using ForwardDiff
using Graphs: AbstractGraph, Edge, SimpleDiGraphFromIterator, topological_sort_by_dfs
using LinearAlgebra: Diagonal, I, UniformScaling, lu!, ldiv!
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

       Jacobians,
       ImpulseResponseMaps,
       getG!

include("shift.jl")
include("blocks.jl")
include("model.jl")
include("irf.jl")

end # module
