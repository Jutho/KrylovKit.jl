# Elementary Givens rotation
using LinearAlgebra: Givens

# for reference: Julia LinearAlgebra.Givens
# immutable Givens{T} <: AbstractRotation{T}
#     i1::Int
#     i2::Int
#     c::T
#     s::T
# end

function LinearAlgebra.rmul!(b::OrthonormalBasis, G::Givens)
    q1, q2 = b[G.i1], b[G.i2]
    q1old = copyto!(similar(q1), q1)
    q1 = axpby!(-conj(G.s), q2, G.c, q1)
    q2 = axpby!(G.s, q1old, G.c, q2)
    return b
end

# New types for discarding or for storing successive Givens transformations
struct NoVecs
end
const novecs = NoVecs()
rmulc!(::NoVecs, ::Any) = novecs
