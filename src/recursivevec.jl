struct RecursiveVec{T<:Union{Tuple,AbstractVector}}
    vecs::T

end
function RecursiveVec(arg1::AbstractVector{T}) where T
    if isbitstype(T)
        return RecursiveVec((arg1,))
    else
        return RecursiveVec{typeof(arg1)}(arg1)
    end
end
RecursiveVec(arg1, args...) = RecursiveVec((arg1, args...))

Base.getindex(v::RecursiveVec, i) = v.vecs[i]

Base.iterate(v::RecursiveVec, args...) = iterate(v.vecs, args...)

Base.IteratorEltype(::Type{RecursiveVec{T}}) where T = Base.IteratorEltype(T)
Base.IteratorSize(::Type{RecursiveVec{T}}) where T = Base.IteratorSize(T)

Base.eltype(v::RecursiveVec) = eltype(v.vecs)
Base.size(v::RecursiveVec) = size(v.vecs)
Base.length(v::RecursiveVec) = length(v.vecs)

Base.first(v::RecursiveVec) = first(v.vecs)
Base.last(v::RecursiveVec) = last(v.vecs)

Base.:-(v::RecursiveVec) = RecursiveVec(map(-,v.vecs))
Base.:+(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(+,v.vecs, w.vecs))
Base.:-(v::RecursiveVec, w::RecursiveVec) = RecursiveVec(map(-,v.vecs, w.vecs))
Base.:*(v::RecursiveVec, a) = RecursiveVec(map(x->x*a, v.vecs))
Base.:*(a, v::RecursiveVec) = RecursiveVec(map(x->a*x, v.vecs))
Base.:/(v::RecursiveVec, a) = RecursiveVec(map(x->x/a, v.vecs))
Base.:\(a, v::RecursiveVec) = RecursiveVec(map(x->a\x, v.vecs))

function Base.similar(v::RecursiveVec)
    RecursiveVec(similar.(v.vecs))
end

function Base.copy!(w::RecursiveVec, v::RecursiveVec)
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
