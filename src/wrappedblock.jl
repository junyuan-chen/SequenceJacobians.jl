struct WrappedBlock{B,J,ins,outs} <: AbstractBlock{ins,outs}
    b::B
    j::J
    WrappedBlock(b::AbstractBlock, j) =
        new{typeof(b),typeof(j),inputs(b),outputs(b)}(b, j)
end

wrap(b::AbstractBlock, j) = WrappedBlock(b, j)

invars(b::WrappedBlock) = invars(b.b)
ssinputs(b::WrappedBlock) = ssinputs(b.b)

outlength(b::WrappedBlock, varvals::NamedTuple) = outlength(b.b, varvals)
outlength(b::WrappedBlock, varvals::NamedTuple, r::Int) = outlength(b.b, varvals, r)

steadystate!(b::WrappedBlock, varvals::NamedTuple) = steadystate!(b.b, varvals)

show(io::IO, b::WrappedBlock) = print(io, "WrappedBlock($(b.b))")

function show(io::IO, ::MIME"text/plain", b::WrappedBlock)
    println(io, "WrappedBlock($(b.b)):")
    _showinouts(io, b)
end
