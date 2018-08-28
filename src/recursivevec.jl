struct RecursiveVec{T<:Tuple}
    vecs::T
end
RecursiveVec(arg1, args...) = RecursiveVec((arg1, args...))

Base.getindex(v::RecursiveVec, i) = v.vecs[i]

Base.iterate(v::RecursiveVec) = iterate(v.vecs)
Base.iterate(v::RecursiveVec, s) = iterate(v.vecs, s)
Base.IteratorEltype(::Type{<:RecursiveVec}) = Base.EltypeUnknown() # since `eltype` is not the eltype of the iterator
Base.IteratorSize(::Type{<:RecursiveVec}) = Base.HasLength()
Base.length(v::RecursiveVec) = length(v.vecs)

Base.first(v::RecursiveVec) = first(v.vecs)
Base.last(v::RecursiveVec) = last(v.vecs)

Base.eltype(v::RecursiveVec) = eltype(typeof(v))
Base.eltype(::Type{RecursiveVec{T}}) where {T<:Tuple} = _eltype(T)

_eltype(::Type{Tuple{T}}) where {T} = eltype(T)
function _eltype(::Type{TT}) where {TT<:Tuple}
    T = eltype(Base.tuple_type_head(TT))
    T2 = _eltype(Base.tuple_type_tail(TT))
    T == T2 ? T : error("all elements of a `RecursiveVec` should have same `eltype`")
end

Base.:-(v::RecursiveVec) = RecursiveVec(map(-,v.vecs))
Base.:+(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(+,v.vecs, w.vecs))
Base.:-(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(-,v.vecs, w.vecs))
Base.:*(v::RecursiveVec, a) = RecursiveVec(map(x->x*a, v.vecs))
Base.:*(a, v::RecursiveVec) = RecursiveVec(map(x->a*x, v.vecs))
Base.:/(v::RecursiveVec, a) = RecursiveVec(map(x->x/a, v.vecs))
Base.:\(a, v::RecursiveVec) = RecursiveVec(map(x->a\x, v.vecs))

function Base.similar(v::RecursiveVec, ::Type{T} = eltype(v)) where {T}
    RecursiveVec(map(x->similar(x,T), v.vecs))
end

function Base.fill!(v::RecursiveVec, a)
    for x in v.vecs
        fill!(x, a)
    end
    return v
end

function Base.copyto!(w::RecursiveVec, v::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        copyto!(w.vecs[i], v.vecs[i])
    end
    return w
end

function LinearAlgebra.mul!(w::RecursiveVec, a, v::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        mul!(w.vecs[i], a, v.vecs[i])
    end
    return w
end

function LinearAlgebra.mul!(w::RecursiveVec, v::RecursiveVec, a)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        mul!(w.vecs[i], v.vecs[i], a)
    end
    return w
end

function LinearAlgebra.rmul!(v::RecursiveVec, a)
    for x in v.vecs
        rmul!(x, a)
    end
    return v
end

function LinearAlgebra.axpy!(a, v::RecursiveVec, w::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        axpy!(a, v.vecs[i], w.vecs[i])
    end
    return w
end
function LinearAlgebra.axpby!(a, v::RecursiveVec, b, w::RecursiveVec)
    @assert length(w.vecs) == length(v.vecs)
    @inbounds for i = 1:length(w.vecs)
        axpby!(a, v.vecs[i], b, w.vecs[i])
    end
    return w
end

LinearAlgebra.dot(v::RecursiveVec{T}, w::RecursiveVec{T}) where {T} = sum(dot.(v.vecs, w.vecs))
LinearAlgebra.norm(v::RecursiveVec) = norm(norm.(v.vecs))
