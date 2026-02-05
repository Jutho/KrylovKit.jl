"""
    v = InnerProductVec(vec, dotf, [normf])

Create a new vector `v` from an existing vector `dotf` with a modified inner product given
by `inner`. The vector `vec`, which can be any type (not necessarily `Vector`) that supports
the basic vector interface required by KrylovKit, is wrapped in a custom struct
`v::InnerProductVec`. All vector space functionality such as addition and multiplication
with scalars (both out of place and in place using `mul!`, `rmul!`, `axpy!` and `axpby!`)
applied to `v` is simply forwarded to `v.vec`. The inner product between vectors
`v1 = InnerProductVec(vec1, dotf)` and `v2 = InnerProductVec(vec2, dotf)` is computed as
`dot(v1, v2) = dotf(v1.vec, v2.vec) = dotf(vec1, vec2)`. The inner product between vectors
with different `dotf` functions is not defined.

By default, the norm of `v::InnerProductVec` is defined as
`norm(v) = sqrt(real(dot(v, v))) = sqrt(real(dotf(vec, vec)))`. However, an optional third
argument `normf` can be provided to define a different norm function, which is useful for
cases like symplectic forms where the inner product is skew-symmetric and cannot define a
norm. In this case, `norm(v) = normf(v.vec)`.

In a (linear) map applied to `v`, the original vector can be obtained as `v.vec` or simply
as `v[]`.
"""
struct InnerProductVec{F, N, T}
    vec::T
    dotf::F
    normf::N
end

InnerProductVec(vec, dotf) = InnerProductVec(vec, dotf, nothing)

Base.:-(v::InnerProductVec) = InnerProductVec(-v.vec, v.dotf, v.normf)
function Base.:+(v::InnerProductVec{F, N}, w::InnerProductVec{F, N}) where {F, N}
    return InnerProductVec(v.vec + w.vec, v.dotf, v.normf)
end
function Base.:-(v::InnerProductVec{F, N}, w::InnerProductVec{F, N}) where {F, N}
    return InnerProductVec(v.vec - w.vec, v.dotf, v.normf)
end
Base.:*(v::InnerProductVec, a::Number) = InnerProductVec(v.vec * a, v.dotf, v.normf)
Base.:*(a::Number, v::InnerProductVec) = InnerProductVec(a * v.vec, v.dotf, v.normf)
Base.:/(v::InnerProductVec, a::Number) = InnerProductVec(v.vec / a, v.dotf, v.normf)
Base.:\(a::Number, v::InnerProductVec) = InnerProductVec(a \ v.vec, v.dotf, v.normf)

function Base.similar(v::InnerProductVec, (::Type{T}) = scalartype(v)) where {T}
    return InnerProductVec(similar(v.vec), v.dotf, v.normf)
end

Base.getindex(v::InnerProductVec) = v.vec

function Base.copy!(w::InnerProductVec{F, N}, v::InnerProductVec{F, N}) where {F, N}
    copy!(w.vec, v.vec)
    return w
end

function LinearAlgebra.mul!(
        w::InnerProductVec{F, N}, a::Number, v::InnerProductVec{F, N}
    ) where {F, N}
    mul!(w.vec, a, v.vec)
    return w
end

function LinearAlgebra.mul!(
        w::InnerProductVec{F, N}, v::InnerProductVec{F, N}, a::Number
    ) where {F, N}
    mul!(w.vec, v.vec, a)
    return w
end

function LinearAlgebra.rmul!(v::InnerProductVec, a::Number)
    rmul!(v.vec, a)
    return v
end

function LinearAlgebra.axpy!(
        a::Number, v::InnerProductVec{F, N}, w::InnerProductVec{F, N}
    ) where {F, N}
    axpy!(a, v.vec, w.vec)
    return w
end
function LinearAlgebra.axpby!(
        a::Number, v::InnerProductVec{F, N}, b, w::InnerProductVec{F, N}
    ) where {F, N}
    axpby!(a, v.vec, b, w.vec)
    return w
end

function LinearAlgebra.dot(v::InnerProductVec{F, N}, w::InnerProductVec{F, N}) where {F, N}
    return v.dotf(v.vec, w.vec)
end

VectorInterface.scalartype(::Type{<:InnerProductVec{F, N, T}}) where {F, N, T} = scalartype(T)

function VectorInterface.zerovector(v::InnerProductVec, T::Type{<:Number})
    return InnerProductVec(zerovector(v.vec, T), v.dotf, v.normf)
end
function VectorInterface.zerovector!(v::InnerProductVec)
    return InnerProductVec(zerovector!(v.vec), v.dotf, v.normf)
end
function VectorInterface.zerovector!!(v::InnerProductVec)
    return InnerProductVec(zerovector!!(v.vec), v.dotf, v.normf)
end

function VectorInterface.scale(v::InnerProductVec, a::Number)
    return InnerProductVec(scale(v.vec, a), v.dotf, v.normf)
end
function VectorInterface.scale!!(v::InnerProductVec, a::Number)
    return InnerProductVec(scale!!(v.vec, a), v.dotf, v.normf)
end
function VectorInterface.scale!(v::InnerProductVec, a::Number)
    scale!(v.vec, a)
    return v
end
function VectorInterface.scale!!(
        w::InnerProductVec{F, N}, v::InnerProductVec{F, N}, a::Number
    ) where {F, N}
    return InnerProductVec(scale!!(w.vec, v.vec, a), w.dotf, w.normf)
end
function VectorInterface.scale!(
        w::InnerProductVec{F, N}, v::InnerProductVec{F, N}, a::Number
    ) where {F, N}
    scale!(w.vec, v.vec, a)
    return w
end

function VectorInterface.add(
        v::InnerProductVec{F, N}, w::InnerProductVec{F, N}, a::Number, b::Number
    ) where {F, N}
    return InnerProductVec(add(v.vec, w.vec, a, b), v.dotf, v.normf)
end
function VectorInterface.add!!(
        v::InnerProductVec{F, N}, w::InnerProductVec{F, N}, a::Number, b::Number
    ) where {F, N}
    return InnerProductVec(add!!(v.vec, w.vec, a, b), v.dotf, v.normf)
end
function VectorInterface.add!(
        v::InnerProductVec{F, N}, w::InnerProductVec{F, N}, a::Number, b::Number
    ) where {F, N}
    add!(v.vec, w.vec, a, b)
    return v
end

function VectorInterface.inner(v::InnerProductVec{F, N}, w::InnerProductVec{F, N}) where {F, N}
    return v.dotf(v.vec, w.vec)
end

VectorInterface.norm(v::InnerProductVec{F, Nothing}) where {F} = sqrt(real(inner(v, v)))
VectorInterface.norm(v::InnerProductVec) = v.normf(v.vec)
