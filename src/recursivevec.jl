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
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        copyto!(w.vecs[i], v.vecs[i])
    end
    return w
end

function LinearAlgebra.mul!(w::RecursiveVec, a::Number, v::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        mul!(w.vecs[i], a, v.vecs[i])
    end
    return w
end

function LinearAlgebra.mul!(w::RecursiveVec, v::RecursiveVec, a::Number)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        mul!(w.vecs[i], v.vecs[i], a)
    end
    return w
end

function LinearAlgebra.rmul!(v::RecursiveVec, a::Number)
    for x in v.vecs
        rmul!(x, a)
    end
    return v
end

function LinearAlgebra.axpy!(a::Number, v::RecursiveVec, w::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        axpy!(a, v.vecs[i], w.vecs[i])
    end
    return w
end
function LinearAlgebra.axpby!(a::Number, v::RecursiveVec, b, w::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        axpby!(a, v.vecs[i], b, w.vecs[i])
    end
    return w
end

LinearAlgebra.dot(v::RecursiveVec{T}, w::RecursiveVec{T}) where {T} =
    sum(dot.(v.vecs, w.vecs))
LinearAlgebra.norm(v::RecursiveVec) = norm(norm.(v.vecs))
