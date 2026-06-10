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

# Usage with SkewOrthogonalizer / Symplectic Arnoldi

In order to use a [`SkewOrthogonalizer`](@ref) algorithm in an
[`ArnoldiFactorization`](@ref), the vector type used must support the
[`symplecticform`](@ref) function. There are two ways to achieve this:

1. **Define `KrylovKit.symplecticform(v, w)`** directly for your vector type. For
   `AbstractVector` types, a default implementation is already provided that computes the
   standard symplectic form `ω(v, w) = Σᵢ (v[2i-1] w[2i] - v[2i] w[2i-1])`.

2. **Wrap your vectors** in `SymplecticFormVec(v, skewf)`, where `skewf(v, w)` computes the
   desired symplectic form between the unwrapped vectors `v` and `w`. This is analogous to
   [`InnerProductVec`](@ref) for custom inner products.

See also [`symplecticform`](@ref), [`SkewOrthogonalizer`](@ref).
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

Compute the symplectic form `ω(v, w)` between two vectors `v` and `w`.

The symplectic form is a skew-symmetric bilinear form, i.e. `ω(v, w) = -ω(w, v)`.

# Methods

For [`SymplecticFormVec`](@ref) vectors `v = SymplecticFormVec(vec, skewf)`, the symplectic
form is computed using the custom function: `symplecticform(v, w) = skewf(v.vec, w.vec)`.

For `AbstractVector` types, a default implementation computes the standard (canonical)
symplectic form:

```math
ω(v, w) = v^T J w = \\sum_i \\bigl(v_{2i-1}\\, w_{2i} - v_{2i}\\, w_{2i-1}\\bigr)
```

For other custom vector types, define a method `KrylovKit.symplecticform(v::MyVec, w::MyVec)`
to enable the use of [`SkewOrthogonalizer`](@ref) algorithms.

See also [`SymplecticFormVec`](@ref), [`SkewOrthogonalizer`](@ref),
[`SymplecticBasis`](@ref).
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
