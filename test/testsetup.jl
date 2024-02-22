module TestSetup

import VectorInterface as VI
using VectorInterface
using LinearAlgebra: LinearAlgebra

# Utility functions
# -----------------
"function for determining the precision of a type"
precision(T::Type{<:Number}) = eps(real(T))^(2 / 3)

"function for comparing sets of eigenvalues"
function ≊(list1::AbstractVector, list2::AbstractVector)
    length(list1) == length(list2) || return false
    n = length(list1)
    ind2 = collect(1:n)
    p = sizehint!(Int[], n)
    for i in 1:n
        j = argmin(abs.(view(list2, ind2) .- list1[i]))
        p = push!(p, ind2[j])
        ind2 = deleteat!(ind2, j)
    end
    return list1 ≈ view(list2, p)
end

# Minimal vector type
# -------------------
"""
    MinimalVec{T<:Number,IP}

Minimal interface for a vector. Can support either in-place assignments or not, depending on
`IP=true` or `IP=false`.
"""
struct MinimalVec{V<:AbstractVector,IP}
    vec::V
    function MinimalVec(v::AbstractVector; inplace::Bool=false)
        return new{typeof(v),inplace}(v)
    end
end

isinplace(::Type{MinimalVec{V,IP}}) where {V,IP} = IP
isinplace(v::MinimalVec) = isinplace(typeof(v))

VI.scalartype(::Type{<:MinimalVec{V}}) where {V} = scalartype(V)

function VI.zerovector(v::MinimalVec, S::Type{<:Number})
    return MinimalVec(zerovector(v.vec, S); inplace=isinplace(v))
end
function VI.zerovector!(v::MinimalVec{V,true}) where {V}
    zerovector!(v.vec)
    return v
end
VI.zerovector!!(v::MinimalVec) = isinplace(v) ? zerovector!(v) : zerovector(v)

function VI.scale(v::MinimalVec, α::Number)
    return MinimalVec(scale(v.vec, α); inplace=isinplace(v))
end
function VI.scale!(v::MinimalVec{V,true}, α::Number) where {V}
    scale!(v.vec, α)
    return v
end
function VI.scale!!(v::MinimalVec, α::Number)
    return isinplace(v) ? scale!(v, α) : scale(v, α)
end
function VI.scale!(w::MinimalVec{V,true}, v::MinimalVec{W,true}, α::Number) where {V,W}
    scale!(w.vec, v.vec, α)
    return w
end
function VI.scale!!(w::MinimalVec, v::MinimalVec, α::Number)
    isinplace(w) && return scale!(w, v, α)
    return MinimalVec(scale!!(copy(w.vec), v.vec, α); inplace=false)
end

function VI.add(y::MinimalVec, x::MinimalVec, α::Number, β::Number)
    return MinimalVec(add(y.vec, x.vec, α, β); inplace=isinplace(y))
end
function VI.add!(y::MinimalVec{W,true}, x::MinimalVec{V,true}, α::Number, β::Number) where {W,V}
    add!(y.vec, x.vec, α, β)
    return y
end
function VI.add!!(y::MinimalVec, x::MinimalVec, α::Number, β::Number)
    return isinplace(y) ? add!(y, x, α, β) : add(y, x, α, β)
end

VI.inner(x::MinimalVec, y::MinimalVec) = inner(x.vec, y.vec)
VI.norm(x::MinimalVec) = LinearAlgebra.norm(x.vec)

end
