struct PackedHessenberg{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    data::V
    n::Int
    function PackedHessenberg{T,V}(data::V, n::Int) where {T,V<:AbstractVector{T}}
        @assert length(data) >= ((n*n + 3*n - 2) >> 1)
        new{T,V}(data, n)
    end
end
PackedHessenberg(data::AbstractVector, n::Int) =
    PackedHessenberg{eltype(data),typeof(data)}(data, n)
Base.size(A::PackedHessenberg) = (A.n,A.n)

function Base.replace_in_print_matrix(A::PackedHessenberg, i::Integer, j::Integer,
                                        s::AbstractString)
    i<=j+1 ? s : Base.replace_with_centered_mark(s)
end

function Base.getindex(A::PackedHessenberg{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    if i > j+1
        return zero(T)
    else
        return A.data[((j*j+j-2) >> 1) + i]
    end
end
function Base.setindex!(A::PackedHessenberg{T}, v, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    if i > j+1 && !iszero(v)
        throw(ReadOnlyMemoryError())
    else
        A.data[((j*j+j-2) >> 1) + i] = v
    end
    return v
end

Base.IndexStyle(::Type{<:PackedHessenberg}) = Base.IndexCartesian()

# TODO: add more methods from the AbstractArray interface
