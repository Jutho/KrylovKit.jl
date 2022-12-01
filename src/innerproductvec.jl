"""
    v = InnerProductVec(vec, dotf)

Create a new vector `v` from an existing vector `dotf` with a modified inner product given
by `inner`. The vector `vec`, which can be any type (not necessarily `Vector`) that supports
the basic vector interface required by KrylovKit, is wrapped in a custom struct
`v::InnerProductVec`. All vector space functionality such as addition and multiplication
with scalars (both out of place and in place using `mul!`, `rmul!`, `axpy!` and `axpby!`)
applied to `v` is simply forwarded to `v.vec`. The inner product between vectors
`v1 = InnerProductVec(vec1, dotf)` and `v2 = InnerProductVec(vec2, dotf)` is computed as
`dot(v1, v2) = dotf(v1.vec, v2.vec) = dotf(vec1, vec2)`. The inner product between vectors
with different `dotf` functions is not defined. Similarly, The norm of `v::InnerProductVec`
is defined as `v = sqrt(real(dot(v, v))) = sqrt(real(dotf(vec, vec)))`.

In a (linear) map applied to `v`, the original vector can be obtained as `v.vec` or simply
as `v[]`.
"""
struct InnerProductVec{F,T}
    vec::T
    dotf::F
end

Base.:-(v::InnerProductVec) = InnerProductVec(-v.vec, v.dotf)
function Base.:+(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return InnerProductVec(v.vec + w.vec, v.dotf)
end
function Base.:-(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return InnerProductVec(v.vec - w.vec, v.dotf)
end
Base.:*(v::InnerProductVec, a::Number) = InnerProductVec(v.vec * a, v.dotf)
Base.:*(a::Number, v::InnerProductVec) = InnerProductVec(a * v.vec, v.dotf)
Base.:/(v::InnerProductVec, a::Number) = InnerProductVec(v.vec / a, v.dotf)
Base.:\(a::Number, v::InnerProductVec) = InnerProductVec(a \ v.vec, v.dotf)

function Base.similar(v::InnerProductVec, ::Type{T}=eltype(v)) where {T}
    return InnerProductVec(similar(v.vec), v.dotf)
end

Base.getindex(v::InnerProductVec) = v.vec

function Base.copy!(w::InnerProductVec{F}, v::InnerProductVec{F}) where {F}
    copy!(w.vec, v.vec)
    return w
end

VectorInterface.scalartype(::Type{<:InnerProductVec{T}}) where {T} = scalartype(T)

function VectorInterface.zerovector(v::InnerProductVec, T::Type{<:Number})
    return InnerProductVec(zerovector(v[], T), v.dotf)
end

function VectorInterface.scale(v::InnerProductVec, a::Number)
    return InnerProductVec(scale(v[], a), v.dotf)
end

function VectorInterface.scale!(v::InnerProductVec, a::Number)
    scale!(v.vec, a)
    return v
end

function VectorInterface.scale!(w::InnerProductVec, v::InnerProductVec, a::Number)
    scale!(w.vec, v.vec, a)
    return w
end

function VectorInterface.add!(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number=1,
                              b::Number=1) where {F}
    add!(v[], w[], a, b)
    return v
end

function VectorInterface.inner(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return v.dotf(v[], w[])
end

VectorInterface.norm(v::InnerProductVec) = sqrt(real(inner(v, v)))
