"""
    v = RecursiveVec(vecs)

Create a new vector `v` from an existing (homogeneous or heterogeneous) list of vectors
`vecs` with one or more elements, represented as a `Tuple` or `AbstractVector`. The elements
of `vecs` can be any type of vectors that are supported by KrylovKit. For a heterogeneous
list, it is best to use a tuple for reasons of type stability, while for a homogeneous list,
either a `Tuple` or a `Vector` can be used. From a mathematical perspectve, `v` represents
the direct sum of the vectors in `vecs`. Scalar multiplication and addition of vectors `v`
acts simultaneously on all elements of `v.vecs`. The inner product corresponds to the sum
of the inner products of the individual vectors in the list `v.vecs`.

The vector `v` also adheres to the iteration syntax, but where it will just produce the
individual vectors in `v.vecs`. Hence, `length(v) = length(v.vecs)`. It can also be indexed,
so that `v[i] = v.vecs[i]`, which can be useful in writing a linear map that acts on `v`.
"""
struct RecursiveVec{T<:Union{Tuple,AbstractVector}}
    vecs::T
end
function RecursiveVec(arg1::AbstractVector{T}) where {T}
    if isbitstype(T)
        return RecursiveVec((arg1,))
    else
        return RecursiveVec{typeof(arg1)}(arg1)
    end
end
RecursiveVec(arg1, args...) = RecursiveVec((arg1, args...))

Base.getindex(v::RecursiveVec, i) = v.vecs[i]

Base.iterate(v::RecursiveVec, args...) = iterate(v.vecs, args...)

Base.IteratorEltype(::Type{RecursiveVec{T}}) where {T} = Base.IteratorEltype(T)
Base.IteratorSize(::Type{RecursiveVec{T}}) where {T} = Base.IteratorSize(T)

Base.eltype(v::RecursiveVec) = eltype(v.vecs)
Base.size(v::RecursiveVec) = size(v.vecs)
Base.length(v::RecursiveVec) = length(v.vecs)

Base.first(v::RecursiveVec) = first(v.vecs)
Base.last(v::RecursiveVec) = last(v.vecs)

Base.:-(v::RecursiveVec) = RecursiveVec(map(-, v.vecs))
Base.:+(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(+, v.vecs, w.vecs))
Base.:-(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(-, v.vecs, w.vecs))
Base.:*(v::RecursiveVec, a::Number) = RecursiveVec(map(x -> x * a, v.vecs))
Base.:*(a::Number, v::RecursiveVec) = RecursiveVec(map(x -> a * x, v.vecs))
Base.:/(v::RecursiveVec, a::Number) = RecursiveVec(map(x -> x / a, v.vecs))
Base.:\(a::Number, v::RecursiveVec) = RecursiveVec(map(x -> a \ x, v.vecs))

function Base.similar(v::RecursiveVec)
    return RecursiveVec(similar.(v.vecs))
end

function Base.copy!(w::RecursiveVec, v::RecursiveVec)
    @assert length(w) == length(v)
    @inbounds for i in 1:length(w)
        copyto!(w[i], v[i])
    end
    return w
end

function LinearAlgebra.dot(v::RecursiveVec{T}, w::RecursiveVec{T}) where {T}
    return sum(dot.(v.vecs, w.vecs))
end

VectorInterface.scalartype(::Type{RecursiveVec{T}}) where {T} = scalartype(T)

function VectorInterface.zerovector(v::RecursiveVec, T::Type{<:Number})
    return RecursiveVec(zerovector(v.vecs, T))
end

function VectorInterface.scale(v::RecursiveVec, a::Number)
    return RecursiveVec(scale(v.vecs, a))
end

function VectorInterface.scale!(v::RecursiveVec, a::Number)
    scale!(v.vecs, a)
    return v
end

function VectorInterface.scale!(w::RecursiveVec, v::RecursiveVec, a::Number)
    scale!(w.vecs, v.vecs, a)
    return w
end

function VectorInterface.scale!!(x::RecursiveVec, a::Number)
    return RecursiveVec(scale!!(x.vecs, a))
end

function VectorInterface.scale!!(w::RecursiveVec,
                                 v::RecursiveVec, a::Number)
    return RecursiveVec(scale!!(w.vecs, v.vecs, a))
end

function VectorInterface.add(w::RecursiveVec, v::RecursiveVec, a::Number=One(),
                             b::Number=One())
    return RecursiveVec(add(w.vecs, v.vecs, a, b))
end

function VectorInterface.add!(w::RecursiveVec, v::RecursiveVec, a::Number=One(),
                              b::Number=One())
    add!(w.vecs, v.vecs, a, b)
    return w
end

function VectorInterface.add!!(w::RecursiveVec, v::RecursiveVec,
                               a::Number=One(),
                               b::Number=One())
    return RecursiveVec(add!!(w.vecs, v.vecs, a, b))
end

function VectorInterface.inner(v::RecursiveVec{T}, w::RecursiveVec{T}) where {T}
    return inner(v.vecs, w.vecs)
end

VectorInterface.norm(v::RecursiveVec) = VectorInterface.norm(v.vecs)
