# Definition of an orthonormal basis
struct OrthonormalBasis{T} <: Basis{T}
    basis::Vector{T}
end
OrthonormalBasis{T}() where {T} = OrthonormalBasis{T}(Vector{T}(undef, 0))

# Iterator methods for OrthonormalBasis
Base.IteratorSize(::Type{<:OrthonormalBasis}) = Base.HasLength()
Base.IteratorEltype(::Type{<:OrthonormalBasis}) = Base.HasEltype()

Base.length(b::OrthonormalBasis) = length(b.basis)
Base.eltype(b::OrthonormalBasis{T}) where {T} = T

Base.iterate(b::OrthonormalBasis) = Base.iterate(b.basis)
Base.iterate(b::OrthonormalBasis, state) = Base.iterate(b.basis, state)

Base.getindex(b::OrthonormalBasis, i) = getindex(b.basis, i)
Base.setindex!(b::OrthonormalBasis, i, q) = setindex!(b.basis, i, q)
Base.firstindex(b::OrthonormalBasis) = firstindex(b.basis)
Base.lastindex(b::OrthonormalBasis) = lastindex(b.basis)

Base.first(b::OrthonormalBasis) = first(b.basis)
Base.last(b::OrthonormalBasis) = last(b.basis)

Base.popfirst!(b::OrthonormalBasis) = popfirst!(b.basis)
Base.pop!(b::OrthonormalBasis) = pop!(b.basis)
Base.push!(b::OrthonormalBasis{T}, q::T) where {T} = (push!(b.basis, q); return b)
Base.empty!(b::OrthonormalBasis) = (empty!(b.basis); return b)
Base.sizehint!(b::OrthonormalBasis, k::Int) = (sizehint!(b.basis, k); return b)
Base.resize!(b::OrthonormalBasis, k::Int) = (resize!(b.basis, k); return b)

# Multiplication methods with OrthonormalBasis
function Base.:*(b::OrthonormalBasis, x::AbstractVector)
    S = promote_type(eltype(first(b)), eltype(x))
    y = similar(first(b), S)
    mul!(y, b, x)
end
# function Base.Ac_mul_B(b::OrthonormalBasis, x)
#     S = promote_type(eltype(first(b)), eltype(x))
#     y = Vector{S}(undef, length(b))
#     Ac_mul_B!(y, b, x)
# end

function LinearAlgebra.mul!(y, b::OrthonormalBasis, x::AbstractVector)
    @assert length(x) <= length(b)

    y = fill!(y, zero(eltype(y)))
    @inbounds for (i, xi) = enumerate(x)
        y = axpy!(xi, b[i], y)
    end
    return y
end
# function Base.Ac_mul_B!(y::AbstractVector, b::OrthonormalBasis, x)
#     @assert length(y) == length(b)
#
#     @inbounds for (i, q) = enumerate(b)
#         y[i] = dot(q, x)
#     end
#     return y
# end

# Orthogonalization of a vector against a given OrthonormalBasis
orthogonalize(v, args...) = orthogonalize!(copy(v), args...)

function orthogonalize!(v::T, b::OrthonormalBasis{T}, alg::Orthogonalizer) where {T}
    S = promote_type(eltype(v), eltype(T))
    c = Vector{S}(undef, length(b))
    orthogonalize!(v, b, c, alg)
end

function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt) where {T}
    for (i, q) = enumerate(b)
        x[i] = dot(q, v)
    end
    for (i, q) = enumerate(b)
        v = axpy!(-x[i], q, v)
    end
    return (v, x)
end
function reorthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt) where {T}
    s = similar(x) ## EXTRA ALLOCATION
    for (i, q) = enumerate(b)
        s[i] = dot(q, v)
        x[i] += s[i]
    end
    for (i, q) = enumerate(b)
        v = axpy!(-s[i], q, v)
    end
    return (v, x)
end
function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt2) where {T}
    (v, x) = orthogonalize!(v, b, x, ClassicalGramSchmidt())
    return reorthogonalize!(v, b, x, ClassicalGramSchmidt())
end
function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, alg::ClassicalGramSchmidtIR) where {T}
    nold = norm(v)
    orthogonalize!(v, b, x, ClassicalGramSchmidt())
    nnew = norm(v)
    while nnew < alg.η*nold
        nold = nnew
        (v,x) = reorthogonalize!(v, b, x, ClassicalGramSchmidt())
        nnew = norm(v)
    end
    return (v, x)
end

function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ModifiedGramSchmidt) where {T}
    for (i, q) = enumerate(b)
        s = dot(q, v)
        v = axpy!(-s, q, v)
        x[i] = s
    end
    return (v, x)
end
function reorthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ModifiedGramSchmidt) where {T}
    for (i, q) = enumerate(b)
        s = dot(q, v)
        v = axpy!(-s, q, v)
        x[i] += s
    end
    return (v, x)
end
function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ModifiedGramSchmidt2) where {T}
    (v, x) = orthogonalize!(v, b, x, ModifiedGramSchmidt())
    return reorthogonalize!(v, b, x, ModifiedGramSchmidt())
end
function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, alg::ModifiedGramSchmidtIR) where {T}
    nold = norm(v)
    (v,x) = orthogonalize!(v, b, x, ModifiedGramSchmidt())
    nnew = norm(v)
    while nnew < alg.η*nold
        nold = nnew
        (v,x) = reorthogonalize!(v, b, x, ModifiedGramSchmidt())
        nnew = norm(v)
    end
    return (v, x)
end

# Orthogonalization of a vector against a given normalized vector
function orthogonalize!(v::T, q::T, alg::Union{ClassicalGramSchmidt,ModifiedGramSchmidt}) where {T}
    s = dot(q,v)
    v = axpy!(-s, q, v)
    return (v, s)
end
function orthogonalize!(v::T, q::T, alg::Union{ClassicalGramSchmidt2,ModifiedGramSchmidt2}) where {T}
    s = dot(q,v)
    v = axpy!(-s, q, v)
    ds = dot(q,v)
    v = axpy!(-ds, q, v)
    return (v, s+ds)
end
function orthogonalize!(v::T, q::T, alg::Union{ClassicalGramSchmidtIR,ModifiedGramSchmidtIR}) where {T}
    nold = norm(v)
    s = dot(q,v)
    v = axpy!(-s, q, v)
    nnew = norm(v)
    while nnew < alg.η*nold
        nold = nnew
        ds = dot(q,v)
        v = axpy!(-ds, q, v)
        s += ds
        nnew = norm(v)
    end
    return (v, s)
end

# Orthonormalization: orthogonalization and normalization
orthonormalize(v, args...) = orthonormalize!(copy(v), args...)

function orthonormalize!(v, args...)
    out = orthogonalize!(v, args...) # out[1] === v
    r = norm(v)
    v = rmul!(v, inv(r))
    return tuple(v, r, Base.tail(out)...)
end
