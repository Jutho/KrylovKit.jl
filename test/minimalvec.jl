using VectorInterface

"""
    MinimalVec{V<:AbstractVector}

Minimal interface for a vector.
"""
struct MinimalVec{V<:AbstractVector}
    vec::V
end

VectorInterface.scalartype(::Type{<:MinimalVec{V}}) where {V} = scalartype(V)

function VectorInterface.zerovector(v::MinimalVec, S::Type{<:Number})
    return MinimalVec(zerovector(v.vec, S))
end
VectorInterface.zerovector!!(v::MinimalVec, S::Type{<:Number}) = zerovector(v, S)

VectorInterface.scale(v::MinimalVec, α::Number) = MinimalVec(scale(v.vec, α))
VectorInterface.scale!!(v::MinimalVec, α::Number) = scale(v, α)

function VectorInterface.add(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    return MinimalVec(add(y.vec, x.vec, α, β))
end
function VectorInterface.add!!(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    return add(y, x, α, β)
end

VectorInterface.inner(x::MinimalVec, y::MinimalVec) = inner(x.vec, y.vec)
VectorInterface.norm(x::MinimalVec) = norm(x.vec)

Base.getindex(v::MinimalVec) = v.vec # for convience, should not interfere

# # minimal interface according to docs
# Base.:*(a::Number, v::MinimalVec) = MinimalVec(a * v[])

# Base.similar(v::MinimalVec) = MinimalVec(similar(v[]))

# LinearAlgebra.axpy!(α, v::MinimalVec, w::MinimalVec) = (axpy!(α, v[], w[]); return w)
# function LinearAlgebra.axpby!(α, v::MinimalVec, β, w::MinimalVec)
#     (axpby!(α, v[], β, w[]); return w)
# end
# LinearAlgebra.rmul!(v::MinimalVec, α) = (rmul!(v[], α); return v)

# LinearAlgebra.mul!(w::MinimalVec, v::MinimalVec, α) = (mul!(w[], v[], α); return w)
# LinearAlgebra.dot(v::MinimalVec, w::MinimalVec) = dot(v[], w[])
# LinearAlgebra.norm(v::MinimalVec) = norm(v[])
