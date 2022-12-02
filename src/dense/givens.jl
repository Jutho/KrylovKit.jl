# Elementary Givens rotation
using LinearAlgebra: Givens

# for reference: Julia LinearAlgebra.Givens
# immutable Givens{T} <: AbstractRotation{T}
#     i1::Int
#     i2::Int
#     c::T
#     s::T
# end

function LinearAlgebra.rmul!(b::OrthonormalBasis{T}, G::Givens) where {T}
    if T isa AbstractArray && IndexStyle(T) isa IndexLinear
        return _rmul_linear!(b, G)
    else
        return _rmul!(b, G)
    end
end

@fastmath function _rmul_linear!(b::OrthonormalBasis{<:AbstractArray}, G::Givens)
    q1, q2 = b[G.i1], b[G.i2]
    c = G.c
    s = G.s
    @inbounds @simd for i in 1:length(q1)
        q1[i], q2[i] = c * q1[i] - conj(s) * q2[i], s * q1[i] + c * q2[i]
    end
    return b
end

function _rmul!(b::OrthonormalBasis, G::Givens)
    q1, q2 = b[G.i1], b[G.i2]
    q1old = scale!!(zerovector(q1), q1, true)
    b[G.i1] = add!!(q1, q2, -conj(G.s), G.c)
    b[G.i2] = add!!(q2, q1old, G.s, G.c)
    return b
end

# New types for discarding or for storing successive Givens transformations
struct NoVecs end
const novecs = NoVecs()
rmulc!(::NoVecs, ::Any) = novecs
