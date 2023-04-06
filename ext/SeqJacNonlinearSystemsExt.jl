module SeqJacNonlinearSystemsExt

if isdefined(Base, :get_extension)
    using NonlinearSystems
    using SequenceJacobians
else
    using ..NonlinearSystems
    using ..SequenceJacobians
end

const NS = NonlinearSystems
const SJ = SequenceJacobians

SJ.isvectorrootsolver(::Type{<:NS.AbstractAlgorithm}) = true
SJ.isvectorrootsolver(::Type{NS.Hybrid}) = true
SJ.isvectorrootsolver(::NS.AbstractAlgorithm) = true

SJ.root(s::NS.NonlinearSystem) = s.x
SJ.rootisfound(s::NS.NonlinearSystem) =
    !(NS.getexitstate(s) ∈ (NS.failed, NS.maxiter_reached))

end # module
