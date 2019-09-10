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
using the functions [`w, = orthogonalize(v,b,...)`](@ref orthogonalize) or [`w, = orthonormalize(v,b,...)`](@ref orthonormalize).
The resulting vector `w` of the latter can then be added to `b` using `push!(b, w)`. Note that
in place versions [`orthogonalize!(v, b, ...)`](@ref orthogonalize) or [`orthonormalize!(v, b, ...)`](@ref orthonormalize)
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
LinearAlgebra.mul!(y, b::OrthonormalBasis, x::AbstractVector) = unproject!(y, b, x, 1, 0)

const BLOCKSIZE = 4096

function project!(y::AbstractVector, b::OrthonormalBasis, x, α::Number = true, β::Number = false, r = Base.OneTo(length(b)))
    # no specialized routine for IndexLinear x because reduction dimension is large dimension
    length(y) == length(r) || throw(DimensionMismatch())
    Threads.@threads for j = 1:length(r)
        @inbounds begin
            if β == 0
                y[j] = α * dot(b[r[j]], x)
            else
                y[j] = β*y[j] + α * dot(b[r[j]], x)
            end
        end
    end
    return y
end

"""
    unproject!(y, b::OrthonormalBasis, x::AbstractVector, α::Number = 1, β::Number = 0, r = Base.OneTo(length(b)))

For a given orthonormal basis `b`, reconstruct the vector-like object `y` that is defined by
expansion coefficients with respect to the basis vectors in `b` in `x`; more specifically
this computes
```
    y = β*y + α * sum(b[r[i]]*x[i] for i = 1:length(r))
```
"""
function unproject!(y, b::OrthonormalBasis, x::AbstractVector, α::Number = true, β::Number = false, r = Base.OneTo(length(b)))
    if y isa AbstractArray && IndexStyle(y) isa IndexLinear && Threads.nthreads() > 1
        return unproject_linear_multithreaded!(y, b, x, α, β, r)
    end
    # general case: using only vector operations, i.e. axpy! (similar to BLAS level 1)
    length(x) == length(r) || throw(DimensionMismatch())
    if β == 0
        fill!(y, zero(eltype(y)))
    elseif β != 1
        rmul!(y, β)
    end
    @inbounds for (i, ri) = enumerate(r)
        y = axpy!(α*x[i], b[ri], y)
    end
    return y
end
function unproject_linear_multithreaded!(y::AbstractArray, b::OrthonormalBasis{<:AbstractArray}, x::AbstractVector, α::Number = true, β::Number = false, r = Base.OneTo(length(b)))
    # multi-threaded implementation, similar to BLAS level 2 matrix vector multiplication
    m = length(y)
    n = length(r)
    length(x) == n || throw(DimensionMismatch())
    for rj in r
        length(b[rj]) == m || throw(DimensionMismatch())
    end
    if n == 0
        return β == 1 ? y : β == 0 ? fill!(y, 0) : rmul!(y, β)
    end
    let m = m, n = n, y = y, x = x, b = b, blocksize = prevpow(2, div(BLOCKSIZE, n))
        Threads.@threads for I = 1:blocksize:m
            unproject_linear_kernel!(y, b, x, I:min(I+blocksize-1, m), α, β, r)
        end
    end
    return y
end
function unproject_linear_kernel!(y::AbstractArray, b::OrthonormalBasis{<:AbstractArray}, x::AbstractVector, I, α::Number, β::Number, r)
    @inbounds begin
        if β == 0
            @simd for i in I
                y[i] = zero(y[i])
            end
        elseif β != 1
            @simd for i in I
                y[i] *= β
            end
        end
        for (j,rj) in enumerate(r)
            xj = x[j]*α
            Vj = b[rj]
            @simd for i in I
                y[i] += Vj[i]*xj
            end
        end
    end
end

"""
    rank1update!(b::OrthonormalBasis, y, x::AbstractVector, α::Number = 1, β::Number = 1, r = Base.OneTo(length(b)))

Perform a rank 1 update of a basis `b`, i.e. update the basis vectors as
```
    b[r[i]] = β*b[r[i]] + α * y * conj(x[i])
```
It is the user's responsibility to make sure that the result is still an orthonormal basis.
"""
@fastmath function rank1update!(b::OrthonormalBasis, y, x::AbstractVector, α::Number = true, β::Number = true, r = Base.OneTo(length(b)))
    if y isa AbstractArray && IndexStyle(y) isa IndexLinear && Threads.nthreads() > 1
        return rank1update_linear_multithreaded!(b, y, x, α, β, r)
    end
    # general case: using only vector operations, i.e. axpy! (similar to BLAS level 1)
    length(x) == length(r) || throw(DimensionMismatch())
    @inbounds for (i, ri) = enumerate(r)
        if β == 1
            b[ri] = axpy!(α*conj(x[i]), y, b[ri])
        elseif β == 0
            b[ri] = mul!(b[ri], α*x[i], y)
        else
            b[ri] = axpby!(α*x[i], y, β, b[ri])
        end
    end
    return b
end
@fastmath function rank1update_linear_multithreaded!(b::OrthonormalBasis{<:AbstractArray}, y::AbstractArray, x::AbstractVector, α::Number, β::Number, r)
    # multi-threaded implementation, similar to BLAS level 2 matrix vector multiplication
    m = length(y)
    n = length(r)
    length(x) == n || throw(DimensionMismatch())
    for rj in r
        length(b[rj]) == m || throw(DimensionMismatch())
    end
    if n == 0
        return b
    end
    blocksize = prevpow(2, div(BLOCKSIZE, n))
    let m = m, n = n, y = y, x = x, b = b, blocksize = prevpow(2, div(BLOCKSIZE, n))
        Threads.@threads for I = 1:blocksize:m
            @inbounds begin
                for (j,rj) in enumerate(r)
                    xj = α*conj(x[j])
                    Vj = b[rj]
                    if β == 0
                        @simd for i = I:min(I+blocksize-1, m)
                            Vj[i] = zero(Vj[i])
                        end
                    elseif β != 1
                        @simd for i = I:min(I+blocksize-1, m)
                            Vj[i] *= β
                        end
                    end
                    if I + blocksize-1 <= m
                        @simd for i = Base.OneTo(blocksize)
                            Vj[I-1+i] += y[I-1+i]*xj
                        end
                    else
                        @simd for i = I:m
                            Vj[i] += y[i]*xj
                        end
                    end
                end
            end
        end
    end
    return b
end

function basistransform!(b::OrthonormalBasis{T}, U::AbstractMatrix) where {T} # U should be unitary or isometric
    if T<:AbstractArray && IndexStyle(T) isa IndexLinear && Threads.nthreads() > 1
        return basistransform_linear_multithreaded!(b, U)
    end
    m, n = size(U)
    m == length(b) || throw(DimensionMismatch())

    b2 = [similar(b[1]) for j = 1:n]
    Threads.@threads for j = 1:n
        mul!(b2[j], b[1], U[1,j])
        for i = 2:m
            axpy!(U[i,j], b[i], b2[j])
        end
    end
    for j = 1:n
        b[j] = b2[j]
    end
    return b
end

function basistransform_linear_multithreaded!(b::OrthonormalBasis{<:AbstractArray}, U::AbstractMatrix) # U should be unitary or isometric
    m, n = size(U)
    m == length(b) || throw(DimensionMismatch())
    K = length(b[1])

    blocksize = prevpow(2, div(BLOCKSIZE, m))
    let b2 = [similar(b[1]) for j = 1:n], K = K, m = m, n = n
        Threads.@threads for I = 1:blocksize:K
            @inbounds for j = 1:n
                b2j = b2[j]
                @simd for i = I:min(I+blocksize-1, K)
                    b2j[i] = zero(b2j[i])
                end
                for k = 1:m
                    bk = b[k]
                    Ukj = U[k,j]
                    @simd for i = I:min(I+blocksize-1, K)
                        b2j[i] += bk[i] * Ukj
                    end
                end
            end
        end
        for j = 1:n
            b[j] = b2[j]
        end
    end
    return b
end

# function basistransform2!(b::OrthonormalBasis, U::AbstractMatrix) # U should be unitary or isometric
#     m, n = size(U)
#     m == length(b) || throw(DimensionMismatch())
#
#     # apply basis transform via householder reflections
#     for j = 1:size(U,2)
#         h, ν = householder(U, j:m, j)
#         lmul!(h, view(U, :, j+1:n))
#         rmul!(b, h')
#     end
#     return b
# end

# Orthogonalization of a vector against a given OrthonormalBasis
orthogonalize(v, args...) = orthogonalize!(copy(v), args...)

function orthogonalize!(v::T, b::OrthonormalBasis{T}, alg::Orthogonalizer) where {T}
    S = promote_type(eltype(v), eltype(T))
    c = Vector{S}(undef, length(b))
    orthogonalize!(v, b, c, alg)
end

function orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt) where {T}
    x = project!(x, b, v)
    v = unproject!(v, b, x, -1, 1)
    return (v, x)
end
function reorthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt) where {T}
    s = similar(x) ## EXTRA ALLOCATION
    s = project!(s, b, v)
    v = unproject!(v, b, s, -1, 1)
    x .+= s
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
