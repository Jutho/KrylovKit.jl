# Definition of an orthonormal basis
"""
    OrthonormalBasis{T} <: Basis{T}

A list of vector like objects of type `T` that are mutually orthogonal and normalized to one,
representing an orthonormal basis for some subspace (typically a Krylov subspace). See also
[`Basis`](@ref)

Orthonormality of the vectors contained in an instance `b` of `OrthonormalBasis`
(i.e. `all(dot(b[i],b[j]) == I[i,j] for i=1:lenght(b), j=1:length(b))`) is not checked when elements are added; it is up to the algorithm that constructs `b` to
guarantee orthonormality.

One can easily orthogonalize or orthonormalize a given vector `v` with respect to a `b::OrthonormalBasis`
using the functions [`w, = orthogonalize(v,b,...)`](@ref orthogonalize) or [`w, = orthonormalize(v,b,...)`](@ref orhonormalize).
The resulting vector `w` of the latter can then be added to `b` using `push!(b, w)`. Note that
in place versions [`orthogonalize!(v, b, ...)`](@ref orthgonalize!) or [`orthonormalize!(v, b, ...)`](@ref orthonormalize!)
are also available.

Finally, a linear combination of the vectors in `b::OrthonormalBasis` can be obtained by multiplying
`b` with a `Vector{<:Number}` using `*` or `mul!` (if the output vector is already allocated).
"""
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
    while eps(one(nnew)) < nnew < alg.η * nold
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
    while eps(one(nnew)) < nnew < alg.η * nold
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
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        ds = dot(q,v)
        v = axpy!(-ds, q, v)
        s += ds
        nnew = norm(v)
    end
    return (v, s)
end

"""
    orthogonalize(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, x
    orthogonalize!(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, x

    orthogonalize(v, q, algorithm::Orthogonalizer]) -> w, s
    orthogonalize!(v, q, algorithm::Orthogonalizer]) -> w, s

Orthogonalize vector `v` against all the vectors in the orthonormal basis `b` using the
orthogonalization algorithm `algorithm`, and return the resulting vector `w` and the overlap
coefficients `x` of `v` with the basis vectors in `b`.

In case of `orthogonalize!`, the vector `v` is mutated in place. In both functions, storage
for the overlap coefficients `x` can be provided as optional argument `x::AbstractVector` with
`length(x) >= length(b)`.

One can also orthogonalize `v` against a given vector `q` (assumed to be normalized), in which
case the orthogonal vector `w` and the inner product `s` between `v` and `q` are returned.

Note that `w` is not normalized, see also [`orthonormalize`](@ref).

For algorithms, see [`ClassicalGramSchmidt`](@ref), [`ModifiedGramSchmidt`](@ref), [`ClassicalGramSchmidt2`](@ref),
[`ModifiedGramSchmidt2`](@ref), [`ClassicalGramSchmidtIR`](@ref) and [`ModifiedGramSchmidtIR`](@ref).
"""
orthogonalize, orthogonalize!

# Orthonormalization: orthogonalization and normalization
orthonormalize(v, args...) = orthonormalize!(copy(v), args...)

function orthonormalize!(v, args...)
    out = orthogonalize!(v, args...) # out[1] === v
    β = norm(v)
    v = rmul!(v, inv(β))
    return tuple(v, β, Base.tail(out)...)
end

"""
    orthonormalize(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, β, x
    orthonormalize!(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, β, x

    orthonormalize(v, q, algorithm::Orthogonalizer]) -> w, β, s
    orthonormalize!(v, q, algorithm::Orthogonalizer]) -> w, β, s

Orthonormalize vector `v` against all the vectors in the orthonormal basis `b` using the
orthogonalization algorithm `algorithm`, and return the resulting vector `w` (of norm 1), its
norm `β` after orthogonalizing and the overlap coefficients `x` of `v` with the basis vectors in
`b`, such that `v = β * w + b * x`.

In case of `orthogonalize!`, the vector `v` is mutated in place. In both functions, storage
for the overlap coefficients `x` can be provided as optional argument `x::AbstractVector` with
`length(x) >= length(b)`.

One can also orthonormalize `v` against a given vector `q` (assumed to be normalized), in which
case the orthonormal vector `w`, its norm `β` before normalizing and the inner product `s` between
`v` and `q` are returned.

See [`orthogonalize`](@ref) if `w` does not need to be normalized.

For algorithms, see [`ClassicalGramSchmidt`](@ref), [`ModifiedGramSchmidt`](@ref), [`ClassicalGramSchmidt2`](@ref),
[`ModifiedGramSchmidt2`](@ref), [`ClassicalGramSchmidtIR`](@ref) and [`ModifiedGramSchmidtIR`](@ref).
"""
orthonormalize, orthonormalize!
