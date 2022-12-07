using VectorInterface

"""
    MinimalVec{T<:Number}

Minimal interface for an out-of-place vector.
"""
struct MinimalVec{T<:Number}
    vec::Vector{T}
end

VectorInterface.scalartype(::Type{MinimalVec{T}}) where {T} = T

function VectorInterface.zerovector(v::MinimalVec, S::Type{<:Number})
    return MinimalVec(zerovector(v.vec, S))
end
VectorInterface.zerovector!!(v::MinimalVec) = zerovector(v)

VectorInterface.scale(v::MinimalVec, α::Number) = MinimalVec(scale(v.vec, α))
VectorInterface.scale!!(v::MinimalVec, α::Number) = scale(v, α)
function VectorInterface.scale!!(w::MinimalVec{T₁}, v::MinimalVec{T₂},
                                 α::Number) where {T₁,T₂}
    return MinimalVec(scale!!(copy(w[]), v[], α))
end

function VectorInterface.add(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    return MinimalVec(add(y.vec, x.vec, α, β))
end
function VectorInterface.add!!(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    return add(y, x, α, β)
end

VectorInterface.inner(x::MinimalVec, y::MinimalVec) = inner(x.vec, y.vec)
VectorInterface.norm(x::MinimalVec) = norm(x.vec)

Base.getindex(v::MinimalVec) = v.vec # for convience, should not interfere
