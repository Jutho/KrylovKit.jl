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

VectorInterface.scalartype(::Type{RecursiveVec{T}}) where {T} = scalartype(eltype(T))

function VectorInterface.zerovector(v::RecursiveVec, T::Type{<:Number})
    return RecursiveVec(zerovector.(v.vecs, T))
end

VectorInterface.scale(v::RecursiveVec, a::Number) = RecursiveVec(scale.(v.vecs, a))

function VectorInterface.scale!(v::RecursiveVec, a::Number)
    @inbounds for i in 1:length(v.vecs)
        scale!(v.vecs[i], a)
    end
    return v
end

function VectorInterface.scale!(w::RecursiveVec, v::RecursiveVec, a::Number)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        scale!(w.vecs[i], v.vecs[i], a)
    end
    return w
end

function VectorInterface.scale!!(x::RecursiveVec{<:Tuple}, a::Number)
    return RecursiveVec(ntuple(i -> scale!!(x[i], a), length(x)))
end

function VectorInterface.scale!!(x::RecursiveVec{<:AbstractVector}, a::Number)
    return RecursiveVec(scale!!.(x.vecs, a))
end

function VectorInterface.add(w::RecursiveVec{T}, v::RecursiveVec{T}, a::ONumber=_one,
                             b::ONumber=_one) where {T<:Tuple}
    return RecursiveVec(ntuple(i -> add(w[i], v[i], a, b), length(w)))
end

function VectorInterface.add(w::RecursiveVec{T}, v::RecursiveVec{T}, a::ONumber=_one,
                             b::ONumber=_one) where {T<:AbstractVector}
    return RecursiveVec(add.(w, v, Ref(a), Ref(b)))
end

function VectorInterface.add!(w::RecursiveVec, v::RecursiveVec, a::ONumber=_one,
                              b::ONumber=_one)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i in 1:length(w.vecs)
        add!(w.vecs[i], v.vecs[i], a, b)
    end
    return w
end

function VectorInterface.add!!(w::RecursiveVec, v::RecursiveVec,
                               a::ONumber=_one,
                               b::ONumber=_one)
    return RecursiveVec(add!!.(w, v, a, b))
end

function VectorInterface.inner(v::RecursiveVec{T}, w::RecursiveVec{T}) where {T}
    return sum(inner.(v.vecs, w.vecs))
end

VectorInterface.norm(v::RecursiveVec) = norm(norm.(v.vecs))
