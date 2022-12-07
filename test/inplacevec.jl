using VectorInterface

"""
    MinimalVec{T<:Number}

Minimal interface for an in-place vector.
"""
struct MinimalVec{T<:Number}
    vec::Vector{T}
end

VectorInterface.scalartype(::Type{MinimalVec{T}}) where {T} = T

function VectorInterface.zerovector(v::MinimalVec, S::Type{<:Number})
    return MinimalVec(zerovector(v.vec, S))
end
function VectorInterface.zerovector!(v::MinimalVec)
    zerovector!(v.vec)
    return v
end
VectorInterface.zerovector!!(v::MinimalVec) = zerovector!(v)

VectorInterface.scale(v::MinimalVec, α::Number) = MinimalVec(scale(v.vec, α))
function VectorInterface.scale!(v::MinimalVec, α::Number)
    scale!(v.vec, α)
    return v
end
VectorInterface.scale!!(v::MinimalVec, α::Number) = scale!(v, α)
function VectorInterface.scale!(w::MinimalVec, v::MinimalVec, α::Number)
    scale!(w.vec, v.vec, α)
    return w
end
VectorInterface.scale!!(w::MinimalVec, v::MinimalVec, α::Number) = scale!(w, v, α)

function VectorInterface.add(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    return MinimalVec(add(y.vec, x.vec, α, β))
end
function VectorInterface.add!(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    add!(y.vec, x.vec, α, β)
    return y
end
function VectorInterface.add!!(y::MinimalVec, x::MinimalVec, α::Number=1, β::Number=1)
    return add!(y, x, α, β)
end

VectorInterface.inner(x::MinimalVec, y::MinimalVec) = inner(x.vec, y.vec)
VectorInterface.norm(x::MinimalVec) = norm(x.vec)

Base.getindex(v::MinimalVec) = v.vec # for convience, should not interfere