# Elementary Givens rotation
import Base.LinAlg: Givens

# for reference: Julia Base.LinAlg.Givens
# immutable Givens{T} <: AbstractRotation{T}
#     i1::Int
#     i2::Int
#     c::T
#     s::T
# end

# Selective application of Givens rotations on matrices
function lmul!(x::AbstractVector, G::Givens)
    x1, x2 = x[G.i1], x[G.i2]
    x[G.i1] =       G.c *x1 + G.s*x2
    x[G.i2] = -conj(G.s)*x1 + G.c*x2
    return x
end
function lmul!(A::AbstractMatrix, G::Givens, cols=indices(A,2))
    @inbounds @simd for j in cols
        a1, a2 = A[G.i1,j], A[G.i2,j]
        A[G.i1,j] =       G.c *a1 + G.s*a2
        A[G.i2,j] = -conj(G.s)*a1 + G.c*a2
    end
    return A
end
function rmulc!(A::AbstractMatrix, G::Givens, rows=indices(A,1))
    @inbounds @simd for i in rows
        a1, a2 = A[i,G.i1], A[i,G.i2]
        A[i,G.i1] =  a1*G.c + a2*conj(G.s)
        A[i,G.i2] = -a1*G.s + a2*G.c
    end
    return A
end
function rmulc!(b::OrthonormalBasis, G::Givens)
    q1, q2 = b[G.i1], b[G.i2]
    q1old = copy(q1)
    axpy!(conj(G.s), q2, scale!(q1, G.c))
    axpy!(-G.s, q1old, scale!(q2, G.c))
    return b
end

# New types for discarding or for storing successive Givens transformations
struct NoVecs
end
const novecs = NoVecs()
rmulc!(::NoVecs, ::Any) = novecs
