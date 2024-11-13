module TestSetup

export tolerance, ≊, MinimalVec, isinplace, stack
export wrapop, wrapvec, unwrapvec, buildrealmap

import VectorInterface as VI
using VectorInterface
using LinearAlgebra: LinearAlgebra

# Utility functions
# -----------------
"function for determining the precision of a type"
tolerance(T::Type{<:Number}) = eps(real(T))^(2 / 3)

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

function buildrealmap(A, B)
    return x -> A * x + B * conj(x)
end

# Minimal vector type
# -------------------
"""
    MinimalVec{T<:Number,IP}

Minimal interface for a vector. Can support either in-place assignments or not, depending on
`IP=true` or `IP=false`.
"""
struct MinimalVec{IP,V<:AbstractVector}
    vec::V
    function MinimalVec{IP}(vec::V) where {IP,V}
        return new{IP,V}(vec)
    end
end
const InplaceVec{V} = MinimalVec{true,V}
const OutplaceVec{V} = MinimalVec{false,V}

isinplace(::Type{MinimalVec{IP,V}}) where {V,IP} = IP
isinplace(v::MinimalVec) = isinplace(typeof(v))

VI.scalartype(::Type{<:MinimalVec{IP,V}}) where {IP,V} = scalartype(V)

function VI.zerovector(v::MinimalVec, S::Type{<:Number})
    return MinimalVec{isinplace(v)}(zerovector(v.vec, S))
end
function VI.zerovector!(v::InplaceVec{V}) where {V}
    zerovector!(v.vec)
    return v
end
VI.zerovector!!(v::MinimalVec) = isinplace(v) ? zerovector!(v) : zerovector(v)

function VI.scale(v::MinimalVec, α::Number)
    return MinimalVec{isinplace(v)}(scale(v.vec, α))
end
function VI.scale!(v::InplaceVec{V}, α::Number) where {V}
    scale!(v.vec, α)
    return v
end
function VI.scale!!(v::MinimalVec, α::Number)
    return isinplace(v) ? scale!(v, α) : scale(v, α)
end
function VI.scale!(w::InplaceVec{V}, v::InplaceVec{W}, α::Number) where {V,W}
    scale!(w.vec, v.vec, α)
    return w
end
function VI.scale!!(w::MinimalVec, v::MinimalVec, α::Number)
    isinplace(w) && return scale!(w, v, α)
    return MinimalVec{false}(scale!!(copy(w.vec), v.vec, α))
end

function VI.add(y::MinimalVec, x::MinimalVec, α::Number, β::Number)
    return MinimalVec{isinplace(y)}(add(y.vec, x.vec, α, β))
end
function VI.add!(y::InplaceVec{W}, x::InplaceVec{V}, α::Number, β::Number) where {W,V}
    add!(y.vec, x.vec, α, β)
    return y
end
function VI.add!!(y::MinimalVec, x::MinimalVec, α::Number, β::Number)
    return isinplace(y) ? add!(y, x, α, β) : add(y, x, α, β)
end

VI.inner(x::MinimalVec, y::MinimalVec) = inner(x.vec, y.vec)
VI.norm(x::MinimalVec) = LinearAlgebra.norm(x.vec)

# Wrappers
# --------
# dispatch on val is necessary for type stability

function wrapvec(v, ::Val{mode}) where {mode}
    return mode === :vector ? v :
           mode === :inplace ? MinimalVec{true}(v) :
           mode === :outplace ? MinimalVec{false}(v) :
           mode === :mixed ? MinimalVec{false}(v) :
           throw(ArgumentError("invalid mode ($mode)"))
end
function wrapvec2(v, ::Val{mode}) where {mode}
    return mode === :mixed ? MinimalVec{true}(v) : wrapvec(v, mode)
end

unwrapvec(v::MinimalVec) = v.vec
unwrapvec(v) = v

function wrapop(A, ::Val{mode}) where {mode}
    if mode === :vector
        return A
    elseif mode === :inplace || mode === :outplace
        return function (v, flag=Val(false))
            if flag === Val(true)
                return wrapvec(A' * unwrapvec(v), Val(mode))
            else
                return wrapvec(A * unwrapvec(v), Val(mode))
            end
        end
    elseif mode === :mixed
        return (x -> wrapvec(A * unwrapvec(x), Val(mode)),
                y -> wrapvec2(A' * unwrapvec(y), Val(mode)))
    else
        throw(ArgumentError("invalid mode ($mode)"))
    end
end

if VERSION < v"1.9"
    stack(f, itr) = mapreduce(f, hcat, itr)
    stack(itr) = reduce(hcat, itr)
end

end
