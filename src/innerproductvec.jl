"""
    v = InnerProductVec(vec, innerf)

Create a new vector `v` from an existing vector `innerf` with a modified inner product given
by `inner`. The vector `vec`, which can be any type (not necessarily `Vector`) that supports
the basic vector interface required by KrylovKit, is wrapped in a custom struct
`v::InnerProductVec`. All vector space functionality such as addition and multiplication
with scalars (both out of place and in place using `mul!`, `rmul!`, `axpy!` and `axpby!`)
applied to `v` is simply forwarded to `v.vec`. The inner product between vectors
`v1 = InnerProductVec(vec1, innerf)` and `v2 = InnerProductVec(vec2, innerf)` is computed as
`inner(v1, v2) = innerf(v1.vec, v2.vec) = innerf(vec1, vec2)`. The inner product between vectors
with different `innerf` functions is not defined. Similarly, The norm of `v::InnerProductVec`
is defined as `v = sqrt(real(inner(v, v))) = sqrt(real(innerf(vec, vec)))`.

In a (linear) map applied to `v`, the original vector can be obtained as `v.vec` or simply
as `v[]`.
"""
struct InnerProductVec{F,T}
    vec::T
    innerf::F
end

# Base.:-(v::InnerProductVec) = InnerProductVec(-v.vec, v.innerf)
# function Base.:+(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
#     return InnerProductVec(v.vec + w.vec, v.innerf)
# end
# function Base.:-(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
#     return InnerProductVec(v.vec - w.vec, v.innerf)
# end
# Base.:*(v::InnerProductVec, a::Number) = InnerProductVec(v.vec * a, v.innerf)
# Base.:*(a::Number, v::InnerProductVec) = InnerProductVec(a * v.vec, v.innerf)
# Base.:/(v::InnerProductVec, a::Number) = InnerProductVec(v.vec / a, v.innerf)
# Base.:\(a::Number, v::InnerProductVec) = InnerProductVec(a \ v.vec, v.innerf)

# function Base.similar(v::InnerProductVec, ::Type{T}=scalartype(v)) where {T}
#     return InnerProductVec(similar(v.vec), v.innerf)
# end

# Base.getindex(v::InnerProductVec) = v.vec

# function Base.copy!(w::InnerProductVec{F}, v::InnerProductVec{F}) where {F}
#     copy!(w.vec, v.vec)
#     return w
# end

# function LinearAlgebra.mul!(w::InnerProductVec{F},
#                             a::Number,
#                             v::InnerProductVec{F}) where {F}
#     mul!(w.vec, a, v.vec)
#     return w
# end

# function LinearAlgebra.mul!(w::InnerProductVec{F},
#                             v::InnerProductVec{F},
#                             a::Number) where {F}
#     mul!(w.vec, v.vec, a)
#     return w
# end

# function LinearAlgebra.rmul!(v::InnerProductVec, a::Number)
#     rmul!(v.vec, a)
#     return v
# end

# function LinearAlgebra.axpy!(a::Number,
#                              v::InnerProductVec{F},
#                              w::InnerProductVec{F}) where {F}
#     axpy!(a, v.vec, w.vec)
#     return w
# end
# function LinearAlgebra.axpby!(a::Number,
#                               v::InnerProductVec{F},
#                               b,
#                               w::InnerProductVec{F}) where {F}
#     axpby!(a, v.vec, b, w.vec)
#     return w
# end

# function LinearAlgebra.dot(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
#     return v.innerf(v.vec, w.vec)
# end

VectorInterface.scalartype(::Type{<:InnerProductVec{F,T}}) where {F,T} = scalartype(T)

function VectorInterface.zerovector(v::InnerProductVec, T::Type{<:Number})
    return InnerProductVec(zerovector(v.vec, T), v.innerf)
end

function VectorInterface.scale(v::InnerProductVec, a::Number)
    return InnerProductVec(scale(v.vec, a), v.innerf)
end
function VectorInterface.scale!!(v::InnerProductVec, a::Number)
    return InnerProductVec(scale!!(v.vec, a), v.innerf)
end
function VectorInterface.scale!(v::InnerProductVec, a::Number)
    scale!(v.vec, a)
    return v
end
function VectorInterface.scale!!(w::InnerProductVec{F}, v::InnerProductVec{F},
                                 a::Number) where {F}
    return InnerProductVec(scale!!(w.vec, v.vec, a), w.innerf)
end
function VectorInterface.scale!(w::InnerProductVec{F}, v::InnerProductVec{F},
                                a::Number) where {F}
    scale!(w.vec, v.vec, a)
    return w
end

function VectorInterface.add(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number,
                             b::Number) where {F}
    return InnerProductVec(add(v.vec, w.vec, a, b), v.innerf)
end
function VectorInterface.add!!(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number,
                               b::Number) where {F}
    return InnerProductVec(add!!(v.vec, w.vec, a, b), v.innerf)
end
function VectorInterface.add!(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number,
                              b::Number) where {F}
    add!(v.vec, w.vec, a, b)
    return v
end

function VectorInterface.inner(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return v.innerf(v.vec, w.vec)
end

VectorInterface.norm(v::InnerProductVec) = sqrt(real(inner(v, v)))
