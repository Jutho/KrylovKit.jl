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

function Base.similar(v::InnerProductVec, ::Type{T}=scalartype(v)) where {T}
    return InnerProductVec(similar(v.vec), v.dotf)
end

function similar_rand(v::InnerProductVec, n::Int)
    @assert n >0
    k = length(v.vec)
    res = [similar(v) for _ in 1:n]
    T = eltype(v.vec)
    try
        res[1].vec .+= rand(T,k)
    catch
        error("Please make sure you have implemented rand operation for your abstract vector type")
    end
    for i in 2:n
        res[i].vec .+= rand(T, k)
    end
    return res
end

function similar_rand(v::AbstractVector,n::Int)
    @assert n >0
    k = length(v)
    res = [similar(v) for _ in 1:n]
    T = eltype(v)
    try
        res[1] .+= rand(T,k)
    catch
        error("Please make sure you have implemented rand operation for your abstract vector type")
    end
    for i in 2:n
        res[i] .+= rand(T, k)
    end
    return res
end


Base.getindex(v::InnerProductVec) = v.vec

function Base.copy!(w::InnerProductVec{F}, v::InnerProductVec{F}) where {F}
    copy!(w.vec, v.vec)
    return w
end

function LinearAlgebra.mul!(w::InnerProductVec{F},
                            a::Number,
                            v::InnerProductVec{F}) where {F}
    mul!(w.vec, a, v.vec)
    return w
end

function LinearAlgebra.mul!(w::InnerProductVec{F},
                            v::InnerProductVec{F},
                            a::Number) where {F}
    mul!(w.vec, v.vec, a)
    return w
end

function LinearAlgebra.rmul!(v::InnerProductVec, a::Number)
    rmul!(v.vec, a)
    return v
end
function LinearAlgebra.mul!(A::AbstractVector{T},B::AbstractVector{T},M::AbstractMatrix) where T
    @inbounds for i in eachindex(A)
        @simd for j in eachindex(B)
            A[i] += B[j] * M[j,i]
        end
    end
    return A
end

function LinearAlgebra.axpy!(a::Number,
                             v::InnerProductVec{F},
                             w::InnerProductVec{F}) where {F}
    axpy!(a, v.vec, w.vec)
    return w
end
function LinearAlgebra.axpby!(a::Number,
                              v::InnerProductVec{F},
                              b,
                              w::InnerProductVec{F}) where {F}
    axpby!(a, v.vec, b, w.vec)
    return w
end

function LinearAlgebra.dot(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return v.dotf(v.vec, w.vec)
end

VectorInterface.scalartype(::Type{<:InnerProductVec{F,T}}) where {F,T} = scalartype(T)

function VectorInterface.zerovector(v::InnerProductVec, T::Type{<:Number})
    return InnerProductVec(zerovector(v.vec, T), v.dotf)
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
function VectorInterface.scale!!(w::InnerProductVec{F}, v::InnerProductVec{F},
                                 a::Number) where {F}
    return InnerProductVec(scale!!(w.vec, v.vec, a), w.dotf)
end
function VectorInterface.scale!(w::InnerProductVec{F}, v::InnerProductVec{F},
                                a::Number) where {F}
    scale!(w.vec, v.vec, a)
    return w
end

function VectorInterface.add(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number,
                             b::Number) where {F}
    return InnerProductVec(add(v.vec, w.vec, a, b), v.dotf)
end
function VectorInterface.add!!(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number,
                               b::Number) where {F}
    return InnerProductVec(add!!(v.vec, w.vec, a, b), v.dotf)
end
function VectorInterface.add!(v::InnerProductVec{F}, w::InnerProductVec{F}, a::Number,
                              b::Number) where {F}
    add!(v.vec, w.vec, a, b)
    return v
end

function VectorInterface.inner(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F}
    return v.dotf(v.vec, w.vec)
end

function blockinner!(M::AbstractMatrix,
    x::AbstractVector,
    y::AbstractVector)
    @assert size(M) == (length(x), length(y)) "Matrix dimensions must match"
    @inbounds for j in eachindex(y)
        yj = y[j]
        for i in eachindex(x)
            M[i, j] = inner(x[i], yj)
        end
    end
    return M
end

VectorInterface.norm(v::InnerProductVec) = sqrt(real(inner(v, v)))

# used for debugging
function blockinner(v::AbstractVector, w::AbstractVector;S::Type = Float64)
    M = Matrix{S}(undef, length(v), length(w))
    blockinner!(M, v, w)
    return M
end

function Base.copyto!(x::InnerProductVec, y::InnerProductVec)
    @assert x.dotf == y.dotf "Dot functions must match"
    copyto!(x.vec, y.vec)
    return x
end


