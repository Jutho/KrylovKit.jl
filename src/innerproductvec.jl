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
struct InnerProductVec{F, T}
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

function Base.similar(v::InnerProductVec, (::Type{T}) = scalartype(v)) where {T}
    return InnerProductVec(similar(v.vec), v.dotf)
end

Base.getindex(v::InnerProductVec) = v.vec

function Base.copy!(w::InnerProductVec{F}, v::InnerProductVec{F}) where {F}
    copy!(w.vec, v.vec)
    return w
end

function LinearAlgebra.mul!(
        w::InnerProductVec{F}, a::Number, v::InnerProductVec{F}
    ) where {F}
    mul!(w.vec, a, v.vec)
    return w
end

function LinearAlgebra.mul!(
        w::InnerProductVec{F}, v::InnerProductVec{F}, a::Number
    ) where {F}
    mul!(w.vec, v.vec, a)
    return w
end

function LinearAlgebra.rmul!(v::InnerProductVec, a::Number)
    rmul!(v.vec, a)
    return v
end

function LinearAlgebra.axpy!(
        a::Number, v::InnerProductVec{F}, w::InnerProductVec{F}
    ) where {F}
    axpy!(a, v.vec, w.vec)
    return w
end
function LinearAlgebra.axpby!(
        a::Number, v::InnerProductVec{F}, b, w::InnerProductVec{F}
    ) where {F}
    axpby!(a, v.vec, b, w.vec)
    return w
end

function LinearAlgebra.dot(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return v.dotf(v.vec, w.vec)
end

VectorInterface.scalartype(::Type{<:InnerProductVec{F, T}}) where {F, T} = scalartype(T)

function VectorInterface.zerovector(v::InnerProductVec, T::Type{<:Number})
    return InnerProductVec(zerovector(v.vec, T), v.dotf)
end
function VectorInterface.zerovector!(v::InnerProductVec)
    return InnerProductVec(zerovector!(v.vec), v.dotf)
end
function VectorInterface.zerovector!!(v::InnerProductVec)
    return InnerProductVec(zerovector!!(v.vec), v.dotf)
end

function VectorInterface.scale(v::InnerProductVec, a::Number)
    return InnerProductVec(scale(v.vec, a), v.dotf)
end
function VectorInterface.scale!!(v::InnerProductVec, a::Number)
    return InnerProductVec(scale!!(v.vec, a), v.dotf)
end
function VectorInterface.scale!(v::InnerProductVec, a::Number)
    scale!(v.vec, a)
    return v
end
function VectorInterface.scale!!(
        w::InnerProductVec{F}, v::InnerProductVec{F}, a::Number
    ) where {F}
    return InnerProductVec(scale!!(w.vec, v.vec, a), w.dotf)
end
function VectorInterface.scale!(
        w::InnerProductVec{F}, v::InnerProductVec{F}, a::Number
    ) where {F}
    scale!(w.vec, v.vec, a)
    return w
end

function VectorInterface.add(
        v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number, b::Number
    ) where {F}
    return InnerProductVec(add(v.vec, w.vec, a, b), v.dotf)
end
function VectorInterface.add!!(
        v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number, b::Number
    ) where {F}
    return InnerProductVec(add!!(v.vec, w.vec, a, b), v.dotf)
end
function VectorInterface.add!(
        v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number, b::Number
    ) where {F}
    add!(v.vec, w.vec, a, b)
    return v
end

function VectorInterface.inner(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return v.dotf(v.vec, w.vec)
end

VectorInterface.norm(v::InnerProductVec) = sqrt(real(inner(v, v)))

"""
    v = SymplecticFormVec(vec, skewf)

Create a new vector `v` from an existing vector `vec` with a custom symplectic form given
by `skewf`. The vector `vec`, which can be any type that supports the basic vector interface
required by KrylovKit, is wrapped in a custom struct `v::SymplecticFormVec`. All vector
space functionality is forwarded to `v.vec`, and the inner product `inner(v1, v2)` is
computed as the standard inner product `inner(v1.vec, v2.vec)`. The symplectic form between
vectors `v1 = SymplecticFormVec(vec1, skewf)` and `v2 = SymplecticFormVec(vec2, skewf)` is
computed as `symplecticform(v1, v2) = skewf(vec1, vec2)`.

In a (linear) map applied to `v`, the original vector can be obtained as `v.vec` or simply
as `v[]`.
"""
struct SymplecticFormVec{F, T}
    vec::T
    skewf::F
end

Base.:-(v::SymplecticFormVec) = SymplecticFormVec(-v.vec, v.skewf)
function Base.:+(v::SymplecticFormVec{F}, w::SymplecticFormVec{F}) where {F}
    return SymplecticFormVec(v.vec + w.vec, v.skewf)
end
function Base.:-(v::SymplecticFormVec{F}, w::SymplecticFormVec{F}) where {F}
    return SymplecticFormVec(v.vec - w.vec, v.skewf)
end
Base.:*(v::SymplecticFormVec, a::Number) = SymplecticFormVec(v.vec * a, v.skewf)
Base.:*(a::Number, v::SymplecticFormVec) = SymplecticFormVec(a * v.vec, v.skewf)
Base.:/(v::SymplecticFormVec, a::Number) = SymplecticFormVec(v.vec / a, v.skewf)
Base.:\(a::Number, v::SymplecticFormVec) = SymplecticFormVec(a \ v.vec, v.skewf)

function Base.similar(v::SymplecticFormVec, (::Type{T}) = scalartype(v)) where {T}
    return SymplecticFormVec(similar(v.vec), v.skewf)
end

Base.getindex(v::SymplecticFormVec) = v.vec

function Base.copy!(w::SymplecticFormVec{F}, v::SymplecticFormVec{F}) where {F}
    copy!(w.vec, v.vec)
    return w
end

function LinearAlgebra.mul!(
        w::SymplecticFormVec{F}, a::Number, v::SymplecticFormVec{F}
    ) where {F}
    mul!(w.vec, a, v.vec)
    return w
end

function LinearAlgebra.mul!(
        w::SymplecticFormVec{F}, v::SymplecticFormVec{F}, a::Number
    ) where {F}
    mul!(w.vec, v.vec, a)
    return w
end

function LinearAlgebra.rmul!(v::SymplecticFormVec, a::Number)
    rmul!(v.vec, a)
    return v
end

function LinearAlgebra.axpy!(
        a::Number, v::SymplecticFormVec{F}, w::SymplecticFormVec{F}
    ) where {F}
    axpy!(a, v.vec, w.vec)
    return w
end
function LinearAlgebra.axpby!(
        a::Number, v::SymplecticFormVec{F}, b, w::SymplecticFormVec{F}
    ) where {F}
    axpby!(a, v.vec, b, w.vec)
    return w
end

function LinearAlgebra.dot(v::SymplecticFormVec{F}, w::SymplecticFormVec{F}) where {F}
    return dot(v.vec, w.vec)
end

VectorInterface.scalartype(::Type{<:SymplecticFormVec{F, T}}) where {F, T} = scalartype(T)

function VectorInterface.zerovector(v::SymplecticFormVec, T::Type{<:Number})
    return SymplecticFormVec(zerovector(v.vec, T), v.skewf)
end
function VectorInterface.zerovector!(v::SymplecticFormVec)
    return SymplecticFormVec(zerovector!(v.vec), v.skewf)
end
function VectorInterface.zerovector!!(v::SymplecticFormVec)
    return SymplecticFormVec(zerovector!!(v.vec), v.skewf)
end

function VectorInterface.scale(v::SymplecticFormVec, a::Number)
    return SymplecticFormVec(scale(v.vec, a), v.skewf)
end
function VectorInterface.scale!!(v::SymplecticFormVec, a::Number)
    return SymplecticFormVec(scale!!(v.vec, a), v.skewf)
end
function VectorInterface.scale!(v::SymplecticFormVec, a::Number)
    scale!(v.vec, a)
    return v
end
function VectorInterface.scale!!(
        w::SymplecticFormVec{F}, v::SymplecticFormVec{F}, a::Number
    ) where {F}
    return SymplecticFormVec(scale!!(w.vec, v.vec, a), w.skewf)
end
function VectorInterface.scale!(
        w::SymplecticFormVec{F}, v::SymplecticFormVec{F}, a::Number
    ) where {F}
    scale!(w.vec, v.vec, a)
    return w
end

function VectorInterface.add(
        v::SymplecticFormVec{F}, w::SymplecticFormVec{F}, a::Number, b::Number
    ) where {F}
    return SymplecticFormVec(add(v.vec, w.vec, a, b), v.skewf)
end
function VectorInterface.add!!(
        v::SymplecticFormVec{F}, w::SymplecticFormVec{F}, a::Number, b::Number
    ) where {F}
    return SymplecticFormVec(add!!(v.vec, w.vec, a, b), v.skewf)
end
function VectorInterface.add!(
        v::SymplecticFormVec{F}, w::SymplecticFormVec{F}, a::Number, b::Number
    ) where {F}
    add!(v.vec, w.vec, a, b)
    return v
end

function VectorInterface.inner(v::SymplecticFormVec{F}, w::SymplecticFormVec{F}) where {F}
    return inner(v.vec, w.vec)
end

VectorInterface.norm(v::SymplecticFormVec) = sqrt(real(inner(v, v)))

# Symplectic form
"""
    symplecticform(v, w)

Compute the symplectic form `ω(v, w)` between two vectors. For `SymplecticFormVec` vectors,
this uses the custom skew-symmetric form provided at construction. For `AbstractVector`,
this computes the standard symplectic form `ω(v, w) = Σ_i (v[2i-1]*w[2i] - v[2i]*w[2i-1])`.
"""
function symplecticform(v::SymplecticFormVec{F}, w::SymplecticFormVec{F}) where {F}
    return v.skewf(v.vec, w.vec)
end

function symplecticform(v::AbstractVector, w::AbstractVector)
    length(v) == length(w) || throw(DimensionMismatch())
    n = length(v)
    T = promote_type(eltype(v), eltype(w))
    result = zero(T)
    @inbounds for i in 1:2:n
        result += v[i] * w[i + 1] - v[i + 1] * w[i]
    end
    return result
end
