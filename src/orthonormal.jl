# Definition of an orthonormal basis
"""
    OrthonormalBasis{T} <: Basis{T}

A list of vector like objects of type `T` that are mutually orthogonal and normalized to
one, representing an orthonormal basis for some subspace (typically a Krylov subspace). See
also [`Basis`](@ref)

Orthonormality of the vectors contained in an instance `b` of `OrthonormalBasis`
(i.e. `all(inner(b[i],b[j]) == I[i,j] for i=1:length(b), j=1:length(b))`) is not checked when
elements are added; it is up to the algorithm that constructs `b` to guarantee
orthonormality.

One can easily orthogonalize or orthonormalize a given vector `v` with respect to a
`b::OrthonormalBasis` using the functions
[`w, = orthogonalize(v,b,...)`](@ref orthogonalize) or
[`w, = orthonormalize(v,b,...)`](@ref orthonormalize). The resulting vector `w` of the
latter can then be added to `b` using `push!(b, w)`. Note that in place versions
[`orthogonalize!!(v, b, ...)`](@ref orthogonalize) or
[`orthonormalize!!(v, b, ...)`](@ref orthonormalize) are also available.

Finally, a linear combination of the vectors in `b::OrthonormalBasis` can be obtained by
multiplying `b` with a `Vector{<:Number}` using `*` or `mul!` (if the output vector is
already allocated).
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
    y = zerovector(first(b), promote_type(scalartype(x), scalartype(first(b))))
    return unproject!!(y, b, x)
end

const BLOCKSIZE = 4096

# helper function to determine if a multithreaded approach should be used
# this uses functionality beyond VectorInterface, but can be faster
_use_multithreaded_array_kernel(y) = _use_multithreaded_array_kernel(typeof(y))
_use_multithreaded_array_kernel(::Type) = false
function _use_multithreaded_array_kernel(::Type{<:Array{T}}) where {T <: Number}
    return isbitstype(T) && get_num_threads() > 1
end
function _use_multithreaded_array_kernel(::Type{<:Basis{T}}) where {T}
    return _use_multithreaded_array_kernel(T)
end

"""
    project!!(y::AbstractVector, b::Basis, x,
        [α::Number = 1, β::Number = 0, r = Base.OneTo(length(b))];
        innerfun = inner)

For a given basis `b`, compute the expansion coefficients `y` resulting from
projecting the vector `x` onto the subspace spanned by `b`; more specifically this computes

```
    y[j] = β*y[j] + α * innerfun(b[r[j]], x)
```

for all ``j ∈ r``. The keyword `innerfun` allows using a custom bilinear form instead of
the default `inner`.
"""
function project!!(
        y::AbstractVector, b::Basis, x,
        α::Number = true, β::Number = false, r = Base.OneTo(length(b));
        innerfun = inner
    )
    # no specialized routine for IndexLinear x because reduction dimension is large dimension
    length(y) == length(r) || throw(DimensionMismatch())
    if get_num_threads() > 1
        @sync for J in splitrange(1:length(r), get_num_threads())
            Threads.@spawn for j in $J
                @inbounds begin
                    if β == 0
                        y[j] = α * innerfun(b[r[j]], x)
                    else
                        y[j] = β * y[j] + α * innerfun(b[r[j]], x)
                    end
                end
            end
        end
    else
        for j in 1:length(r)
            @inbounds begin
                if β == 0
                    y[j] = α * innerfun(b[r[j]], x)
                else
                    y[j] = β * y[j] + α * innerfun(b[r[j]], x)
                end
            end
        end
    end
    return y
end

"""
    unproject!!(y, b::Basis, x::AbstractVector,
        [α::Number = 1, β::Number = 0, r = Base.OneTo(length(b))])

For a given basis `b`, reconstruct the vector-like object `y` that is defined by
expansion coefficients with respect to the basis vectors in `b` in `x`; more specifically
this computes

```
    y = β*y + α * sum(b[r[i]]*x[i] for i = 1:length(r))
```
"""
function unproject!!(
        y, b::Basis, x::AbstractVector,
        α::Number = true, β::Number = false, r = Base.OneTo(length(b))
    )
    if _use_multithreaded_array_kernel(y)
        return unproject_linear_multithreaded!(y, b, x, α, β, r)
    end
    # general case: using only vector operations, i.e. axpy! (similar to BLAS level 1)
    length(x) == length(r) || throw(DimensionMismatch())
    if β == 0
        y = scale!!(y, false) # should be hard zero
    elseif β != 1
        y = scale!!(y, β)
    end
    @inbounds for (i, ri) in enumerate(r)
        y = add!!(y, b[ri], α * x[i])
    end
    return y
end
function unproject_linear_multithreaded!(
        y::AbstractArray, b::Basis{<:AbstractArray}, x::AbstractVector,
        α::Number = true, β::Number = false, r = Base.OneTo(length(b))
    )
    # multi-threaded implementation, similar to BLAS level 2 matrix vector multiplication
    m = length(y)
    n = length(r)
    length(x) == n || throw(DimensionMismatch())
    for rj in r
        length(b[rj]) == m || throw(DimensionMismatch())
    end
    if n == 0
        return β == 1 ? y : β == 0 ? zerovector!(y) : scale!(y, β)
    end
    let m = m, n = n, y = y, x = x, b = b, blocksize = prevpow(2, div(BLOCKSIZE, n))
        @sync for II in splitrange(1:blocksize:m, get_num_threads())
            Threads.@spawn for I in $II
                unproject_linear_kernel!(y, b, x, I:min(I + blocksize - 1, m), α, β, r)
            end
        end
    end
    return y
end
function unproject_linear_kernel!(
        y::AbstractArray, b::Basis{<:AbstractArray}, x::AbstractVector,
        I, α::Number, β::Number, r
    )
    return @inbounds begin
        if β == 0
            @simd for i in I
                y[i] = zero(y[i])
            end
        elseif β != 1
            @simd for i in I
                y[i] *= β
            end
        end
        for (j, rj) in enumerate(r)
            xj = x[j] * α
            Vj = b[rj]
            @simd for i in I
                y[i] += Vj[i] * xj
            end
        end
    end
end

"""
    rank1update!(b::OrthonormalBasis, y, x::AbstractVector,
        [α::Number = 1, β::Number = 1, r = Base.OneTo(length(b))])

Perform a rank 1 update of a basis `b`, i.e. update the basis vectors as

```
    b[r[i]] = β*b[r[i]] + α * y * conj(x[i])
```

It is the user's responsibility to make sure that the result is still an orthonormal basis.
"""
@fastmath function rank1update!(
        b::OrthonormalBasis, y, x::AbstractVector,
        α::Number = true, β::Number = true, r = Base.OneTo(length(b))
    )
    if _use_multithreaded_array_kernel(y)
        return rank1update_linear_multithreaded!(b, y, x, α, β, r)
    end
    # general case: using only vector operations, i.e. axpy! (similar to BLAS level 1)
    length(x) == length(r) || throw(DimensionMismatch())
    @inbounds for (i, ri) in enumerate(r)
        if β == 1
            b[ri] = add!!(b[ri], y, α * conj(x[i]))
        elseif β == 0
            b[ri] = scale!!(b[ri], y, α * conj(x[i]))
        else
            b[ri] = add!!(b[ri], y, α * conj(x[i]), β)
        end
    end
    return b
end
@fastmath function rank1update_linear_multithreaded!(
        b::OrthonormalBasis{<:AbstractArray}, y::AbstractArray, x::AbstractVector,
        α::Number, β::Number, r
    )
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
    let m = m, n = n, y = y, x = x, b = b, blocksize = prevpow(2, div(BLOCKSIZE, n))
        @sync for II in splitrange(1:blocksize:m, get_num_threads())
            Threads.@spawn for I in $II
                @inbounds begin
                    for (j, rj) in enumerate(r)
                        xj = α * conj(x[j])
                        Vj = b[rj]
                        if β == 0
                            @simd for i in I:min(I + blocksize - 1, m)
                                Vj[i] = zero(Vj[i])
                            end
                        elseif β != 1
                            @simd for i in I:min(I + blocksize - 1, m)
                                Vj[i] *= β
                            end
                        end
                        if I + blocksize - 1 <= m
                            @simd for i in Base.OneTo(blocksize)
                                Vj[I - 1 + i] += y[I - 1 + i] * xj
                            end
                        else
                            @simd for i in I:m
                                Vj[i] += y[i] * xj
                            end
                        end
                    end
                end
            end
        end
    end
    return b
end

"""
    basistransform!(b::OrthonormalBasis, U::AbstractMatrix)

Transform the orthonormal basis `b` by the matrix `U`. For `b` an orthonormal basis,
the matrix `U` should be real orthogonal or complex unitary; it is up to the user to ensure
this condition is satisfied. The new basis vectors are given by

```
    b[j] ← b[i] * U[i,j]
```

and are stored in `b`, so the old basis vectors are thrown away. Note that, by definition,
the subspace spanned by these basis vectors is exactly the same.
"""
function basistransform!(b::OrthonormalBasis{T}, U::AbstractMatrix) where {T} # U should be unitary or isometric
    if _use_multithreaded_array_kernel(b)
        return basistransform_linear_multithreaded!(b, U)
    end
    m, n = size(U)
    m == length(b) || throw(DimensionMismatch())

    let b2 = [zerovector(b[1]) for j in 1:n]
        if get_num_threads() > 1
            @sync for J in splitrange(1:n, get_num_threads())
                Threads.@spawn for j in $J
                    b2[j] = scale!!(b2[j], b[1], U[1, j])
                    for i in 2:m
                        b2[j] = add!!(b2[j], b[i], U[i, j])
                    end
                end
            end
        else
            for j in 1:n
                b2[j] = scale!!(b2[j], b[1], U[1, j])
                for i in 2:m
                    b2[j] = add!!(b2[j], b[i], U[i, j])
                end
            end
        end
        for j in 1:n
            b[j] = b2[j]
        end
    end
    return b
end

function basistransform_linear_multithreaded!(
        b::OrthonormalBasis{<:AbstractArray}, U::AbstractMatrix
    ) # U should be unitary or isometric
    m, n = size(U)
    m == length(b) || throw(DimensionMismatch())
    K = length(b[1])

    blocksize = prevpow(2, div(BLOCKSIZE, m))
    let b2 = [similar(b[1]) for j in 1:n], K = K, m = m, n = n
        @sync for II in splitrange(1:blocksize:K, get_num_threads())
            Threads.@spawn for I in $II
                @inbounds for j in 1:n
                    b2j = b2[j]
                    @simd for i in I:min(I + blocksize - 1, K)
                        b2j[i] = zero(b2j[i])
                    end
                    for k in 1:m
                        bk = b[k]
                        Ukj = U[k, j]
                        @simd for i in I:min(I + blocksize - 1, K)
                            b2j[i] += bk[i] * Ukj
                        end
                    end
                end
            end
        end
        for j in 1:n
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
orthogonalize(v, args...) = orthogonalize!!(scale(v, true), args...)

function orthogonalize!!(v::T, b::OrthonormalBasis{T}, alg::Orthogonalizer) where {T}
    S = promote_type(scalartype(v), scalartype(T))
    c = Vector{S}(undef, length(b))
    return orthogonalize!!(v, b, c, alg)
end

function orthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt
    ) where {T}
    x = project!!(x, b, v)
    v = unproject!!(v, b, x, -1, 1)
    return (v, x)
end
function reorthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt
    ) where {T}
    s = similar(x) ## EXTRA ALLOCATION
    s = project!!(s, b, v)
    v = unproject!!(v, b, s, -1, 1)
    x .+= s
    return (v, x)
end
function orthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ClassicalGramSchmidt2
    ) where {T}
    (v, x) = orthogonalize!!(v, b, x, ClassicalGramSchmidt())
    return reorthogonalize!!(v, b, x, ClassicalGramSchmidt())
end
function orthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, alg::ClassicalGramSchmidtIR
    ) where {T}
    nold = norm(v)
    (v, x) = orthogonalize!!(v, b, x, ClassicalGramSchmidt())
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        (v, x) = reorthogonalize!!(v, b, x, ClassicalGramSchmidt())
        nnew = norm(v)
    end
    return (v, x)
end

function orthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ModifiedGramSchmidt
    ) where {T}
    for (i, q) in enumerate(b)
        s = inner(q, v)
        v = add!!(v, q, -s)
        x[i] = s
    end
    return (v, x)
end
function reorthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ModifiedGramSchmidt
    ) where {T}
    for (i, q) in enumerate(b)
        s = inner(q, v)
        v = add!!(v, q, -s)
        x[i] += s
    end
    return (v, x)
end
function orthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, ::ModifiedGramSchmidt2
    ) where {T}
    (v, x) = orthogonalize!!(v, b, x, ModifiedGramSchmidt())
    return reorthogonalize!!(v, b, x, ModifiedGramSchmidt())
end
function orthogonalize!!(
        v::T, b::OrthonormalBasis{T}, x::AbstractVector, alg::ModifiedGramSchmidtIR
    ) where {T}
    nold = norm(v)
    (v, x) = orthogonalize!!(v, b, x, ModifiedGramSchmidt())
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        (v, x) = reorthogonalize!!(v, b, x, ModifiedGramSchmidt())
        nnew = norm(v)
    end
    return (v, x)
end

# Orthogonalization of a vector against a given normalized vector
orthogonalize!!(v::T, q::T, alg::Orthogonalizer) where {T} = _orthogonalize!!(v, q, alg)
# avoid method ambiguity on Julia 1.0 according to Aqua.jl

function _orthogonalize!!(
        v::T, q::T, alg::Union{ClassicalGramSchmidt, ModifiedGramSchmidt}
    ) where {T}
    s = inner(q, v)
    v = add!!(v, q, -s)
    return (v, s)
end
function _orthogonalize!!(
        v::T, q::T, alg::Union{ClassicalGramSchmidt2, ModifiedGramSchmidt2}
    ) where {T}
    s = inner(q, v)
    v = add!!(v, q, -s)
    ds = inner(q, v)
    v = add!!(v, q, -ds)
    return (v, s + ds)
end
function _orthogonalize!!(
        v::T, q::T, alg::Union{ClassicalGramSchmidtIR, ModifiedGramSchmidtIR}
    ) where {T}
    nold = norm(v)
    s = inner(q, v)
    v = add!!(v, q, -s)
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        ds = inner(q, v)
        v = add!!(v, q, -ds)
        s += ds
        nnew = norm(v)
    end
    return (v, s)
end

"""
    orthogonalize(v, b::OrthonormalBasis, [x::AbstractVector,] alg::Orthogonalizer]) -> w, x
    orthogonalize!!(v, b::OrthonormalBasis, [x::AbstractVector,] alg::Orthogonalizer]) -> w, x

    orthogonalize(v, q, algorithm::Orthogonalizer]) -> w, s
    orthogonalize!!(v, q, algorithm::Orthogonalizer]) -> w, s

Orthogonalize vector `v` against all the vectors in the orthonormal basis `b` using the
orthogonalization algorithm `alg` of type [`Orthogonalizer`](@ref), and return the resulting
vector `w` and the overlap coefficients `x` of `v` with the basis vectors in `b`.

In case of `orthogonalize!`, the vector `v` is mutated in place. In both functions, storage
for the overlap coefficients `x` can be provided as optional argument `x::AbstractVector`
with `length(x) >= length(b)`.

One can also orthogonalize `v` against a given vector `q` (assumed to be normalized), in
which case the orthogonal vector `w` and the inner product `s` between `v` and `q` are
returned.

Note that `w` is not normalized, see also [`orthonormalize`](@ref).

For more information on possible orthogonalization algorithms, see [`Orthogonalizer`](@ref)
and its concrete subtypes [`ClassicalGramSchmidt`](@ref), [`ModifiedGramSchmidt`](@ref),
[`ClassicalGramSchmidt2`](@ref), [`ModifiedGramSchmidt2`](@ref),
[`ClassicalGramSchmidtIR`](@ref) and [`ModifiedGramSchmidtIR`](@ref).
"""
orthogonalize, orthogonalize!!

# Orthonormalization: orthogonalization and normalization
orthonormalize(v, args...) = orthonormalize!!(scale(v, VectorInterface.One()), args...)

function orthonormalize!!(v, args...)
    out = orthogonalize!!(v, args...) # out[1] === v
    β = norm(v)
    v = scale!!(v, inv(β))
    return tuple(v, β, Base.tail(out)...)
end

"""
    orthonormalize(v, b::OrthonormalBasis, [x::AbstractVector,] alg::Orthogonalizer]) -> w, β, x
    orthonormalize!!(v, b::OrthonormalBasis, [x::AbstractVector,] alg::Orthogonalizer]) -> w, β, x

    orthonormalize(v, q, algorithm::Orthogonalizer]) -> w, β, s
    orthonormalize!!(v, q, algorithm::Orthogonalizer]) -> w, β, s

Orthonormalize vector `v` against all the vectors in the orthonormal basis `b` using the
orthogonalization algorithm `alg` of type [`Orthogonalizer`](@ref), and return the resulting
vector `w` (of norm 1), its norm `β` after orthogonalizing and the overlap coefficients `x`
of `v` with the basis vectors in `b`, such that `v = β * w + b * x`.

In case of `orthogonalize!`, the vector `v` is mutated in place. In both functions, storage
for the overlap coefficients `x` can be provided as optional argument `x::AbstractVector`
with `length(x) >= length(b)`.

One can also orthonormalize `v` against a given vector `q` (assumed to be normalized), in
which case the orthonormal vector `w`, its norm `β` before normalizing and the inner product
`s` between `v` and `q` are returned.

See [`orthogonalize`](@ref) if `w` does not need to be normalized.

For more information on possible orthogonalization algorithms, see [`Orthogonalizer`](@ref)
and its concrete subtypes [`ClassicalGramSchmidt`](@ref), [`ModifiedGramSchmidt`](@ref),
[`ClassicalGramSchmidt2`](@ref), [`ModifiedGramSchmidt2`](@ref),
[`ClassicalGramSchmidtIR`](@ref) and [`ModifiedGramSchmidtIR`](@ref).
"""
orthonormalize, orthonormalize!!

# Definition of a symplectic (Darboux) basis
"""
    SymplecticBasis{T} <: Basis{T}

A list of vector like objects of type `T` that form a symplectic (Darboux) basis with
respect to a skew-symmetric bilinear form `ω`. A symplectic basis satisfies the relations
`ω(u_{2m-1}, u_{2n}) = δ_{mn}`, `ω(u_{2m}, u_{2n}) = 0`, and `ω(u_{2m-1}, u_{2n-1}) = 0`.
See also [`Basis`](@ref).

Skew-orthonormality of the vectors contained in an instance `b` of `SymplecticBasis` is not
checked when elements are added; it is up to the algorithm that constructs `b` to guarantee
skew-orthonormality.

Vectors are added in pairs: odd-indexed vectors `u_{2m-1}` and their symplectic partners
`u_{2m}`. The function [`skeworthogonalize`](@ref) or [`skeworthonormalize`](@ref) can be
used to skew-orthogonalize a new vector with respect to the existing basis. These functions
require a skew-symmetric form `ω` to be provided. Note that in-place versions
[`skeworthogonalize!!`](@ref) or [`skeworthonormalize!!`](@ref) are also available.
"""
struct SymplecticBasis{T} <: Basis{T}
    basis::Vector{T}
end
SymplecticBasis{T}() where {T} = SymplecticBasis{T}(Vector{T}(undef, 0))

# Iterator methods for SymplecticBasis
Base.IteratorSize(::Type{<:SymplecticBasis}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SymplecticBasis}) = Base.HasEltype()

Base.length(b::SymplecticBasis) = length(b.basis)
Base.eltype(b::SymplecticBasis{T}) where {T} = T

Base.iterate(b::SymplecticBasis) = Base.iterate(b.basis)
Base.iterate(b::SymplecticBasis, state) = Base.iterate(b.basis, state)

Base.getindex(b::SymplecticBasis, i) = getindex(b.basis, i)
Base.setindex!(b::SymplecticBasis, i, q) = setindex!(b.basis, i, q)
Base.firstindex(b::SymplecticBasis) = firstindex(b.basis)
Base.lastindex(b::SymplecticBasis) = lastindex(b.basis)

Base.first(b::SymplecticBasis) = first(b.basis)
Base.last(b::SymplecticBasis) = last(b.basis)

Base.popfirst!(b::SymplecticBasis) = popfirst!(b.basis)
Base.pop!(b::SymplecticBasis) = pop!(b.basis)
Base.push!(b::SymplecticBasis{T}, q::T) where {T} = (push!(b.basis, q); return b)
Base.empty!(b::SymplecticBasis) = (empty!(b.basis); return b)
Base.sizehint!(b::SymplecticBasis, k::Int) = (sizehint!(b.basis, k); return b)
Base.resize!(b::SymplecticBasis, k::Int) = (resize!(b.basis, k); return b)

numpairs(b::SymplecticBasis) = div(length(b), 2)

# Skew-orthogonalization of a vector against a given SymplecticBasis
skeworthogonalize(v, args...) = skeworthogonalize!!(scale(v, true), args...)

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, alg::SkewOrthogonalizer
    ) where {T}
    S = promote_type(scalartype(v), scalartype(T))
    c = Vector{S}(undef, length(b))
    return skeworthogonalize!!(v, b, c, alg)
end

# See pages 3-4 of https://people.math.ethz.ch/%7Eacannas/Papers/lsg.pdf
function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidt
    ) where {T}
    np = numpairs(b)
    idx_odd = 1:2:(2np - 1)
    idx_even = 2:2:2np
    project!!(view(x, idx_odd), b, v, -1, 0, idx_even; innerfun = symplecticform)
    project!!(view(x, idx_even), b, v, 1, 0, idx_odd; innerfun = symplecticform)
    if isodd(length(b))
        if alg.esr == ESR2
            x[2np + 1] = inner(last(b), v)
        else
            x[2np + 1] = zero(eltype(x))
        end
    end
    v = unproject!!(v, b, x, -1, 1)
    return (v, x)
end

function reskeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidt
    ) where {T}
    s = similar(x) ## EXTRA ALLOCATION
    (v, s) = skeworthogonalize!!(v, b, s, ClassicalSymplecticGramSchmidt())
    x .+= s
    return (v, x)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidt2
    ) where {T}
    csgs = ClassicalSymplecticGramSchmidt(alg.esr)
    (v, x) = skeworthogonalize!!(v, b, x, csgs)
    return reskeworthogonalize!!(v, b, x, csgs)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidtIR
    ) where {T}
    csgs = ClassicalSymplecticGramSchmidt(alg.esr)
    nold = norm(v)
    (v, x) = skeworthogonalize!!(v, b, x, csgs)
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        (v, x) = reskeworthogonalize!!(v, b, x, csgs)
        nnew = norm(v)
    end
    return (v, x)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidt
    ) where {T}
    np = numpairs(b)
    for m in 1:np
        i_odd = 2m - 1
        i_even = 2m
        h_e = symplecticform(b[i_odd], v)
        h_f = symplecticform(b[i_even], v)
        x[i_odd] = -h_f
        x[i_even] = h_e
        v = add!!(v, b[i_odd], h_f)
        v = add!!(v, b[i_even], -h_e)
    end
    if isodd(length(b))
        if alg.esr == ESR2
            x[2np + 1] = inner(last(b), v)
            v = add!!(v, last(b), -x[2np + 1])
        else
            x[2np + 1] = zero(eltype(x))
        end
    end
    return (v, x)
end

function reskeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidt
    ) where {T}
    np = numpairs(b)
    for m in 1:np
        i_odd = 2m - 1
        i_even = 2m
        h_e = symplecticform(b[i_odd], v)
        h_f = symplecticform(b[i_even], v)
        x[i_odd] -= h_f
        x[i_even] += h_e
        v = add!!(v, b[i_odd], h_f)
        v = add!!(v, b[i_even], -h_e)
    end
    if isodd(length(b)) && alg.esr == ESR2
        r11 = inner(last(b), v)
        x[2np + 1] += r11
        v = add!!(v, last(b), -r11)
    end
    return (v, x)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidt2
    ) where {T}
    msgs = ModifiedSymplecticGramSchmidt(alg.esr)
    (v, x) = skeworthogonalize!!(v, b, x, msgs)
    return reskeworthogonalize!!(v, b, x, msgs)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidtIR
    ) where {T}
    msgs = ModifiedSymplecticGramSchmidt(alg.esr)
    nold = norm(v)
    (v, x) = skeworthogonalize!!(v, b, x, msgs)
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        (v, x) = reskeworthogonalize!!(v, b, x, msgs)
        nnew = norm(v)
    end
    return (v, x)
end

# Skew-orthonormalization: skew-orthogonalization and normalization
# For odd vectors: normalize with standard norm
# For even vectors: scale so that ω(partner, v) = 1
skeworthonormalize(v, args...) = skeworthonormalize!!(scale(v, VectorInterface.One()), args...)

function skeworthonormalize!!(v, b::SymplecticBasis, x::AbstractVector, alg::SkewOrthogonalizer)
    out = skeworthogonalize!!(v, b, x, alg)
    if iseven(length(b))
        # Adding odd vector: normalize with standard norm
        β = alg.esr == ESR3m ? one(scalartype(v)) : norm(v)
        v = scale!!(v, inv(β))
    else
        # Adding even vector: scale so that ω(partner, v) = 1
        # The partner is the last vector in b (the odd vector of the current pair)
        β = symplecticform(last(b), v)
        v = scale!!(v, inv(β))
    end
    return (v, β, Base.tail(out)...)
end

"""
    skeworthogonalize(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, x
    skeworthogonalize!!(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, x

Skew-orthogonalize vector `v` against all the vectors in the symplectic basis `b` using the
skew-orthogonalization algorithm `alg` of type [`SkewOrthogonalizer`](@ref), and return the
resulting vector `w` and the overlap coefficients `x` of `v` with the basis vectors in `b`.

The skew-orthogonalization uses the symplectic form [`symplecticform`](@ref), which is
expected to satisfy `ω(u, v) = -ω(v, u)`. For vectors wrapped in `SymplecticFormVec`, a
custom form function can be provided. For a symplectic basis with pairs `(u_{2m-1}, u_{2m})`
where `ω(u_{2m-1}, u_{2m}) = 1`, the skew-orthogonalization ensures:
- `ω(u_{2m-1}, w) = 0` for all `m`
- `ω(u_{2m}, w) = 0` for all `m`

In case of `skeworthogonalize!!`, the vector `v` is mutated in place. In both functions,
storage for the overlap coefficients `x` can be provided as optional argument
`x::AbstractVector` with `length(x) >= length(b)`.

Note that `w` is not normalized, see also [`skeworthonormalize`](@ref).

For more information on possible skew-orthogonalization algorithms, see
[`SkewOrthogonalizer`](@ref) and its concrete subtypes
[`ClassicalSymplecticGramSchmidt`](@ref), [`ModifiedSymplecticGramSchmidt`](@ref),
[`ClassicalSymplecticGramSchmidt2`](@ref), [`ModifiedSymplecticGramSchmidt2`](@ref),
[`ClassicalSymplecticGramSchmidtIR`](@ref) and [`ModifiedSymplecticGramSchmidtIR`](@ref).
"""
skeworthogonalize, skeworthogonalize!!

"""
    skeworthonormalize(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, β, x
    skeworthonormalize!!(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, β, x

Skew-orthonormalize vector `v` against all the vectors in the symplectic basis `b` using
the skew-orthogonalization algorithm `alg` of type [`SkewOrthogonalizer`](@ref).

The normalization depends on the current length of the basis:

**When `length(b)` is even** (adding an odd vector): returns the resulting vector `w`
normalized to unit norm (`‖w‖ = 1`), the norm `β = ‖v‖` after skew-orthogonalizing, and
the overlap coefficients `x`.

**When `length(b)` is odd** (adding an even vector): returns the resulting vector `w`
scaled such that `ω(last(b), w) = 1`, where `last(b)` is the odd partner of the pair.
Returns the scaling factor `β = ω(last(b), v)` after skew-orthogonalizing, and the overlap
coefficients `x`.

In case of `skeworthonormalize!!`, the vector `v` is mutated in place. In both functions,
storage for the overlap coefficients `x` can be provided as optional argument
`x::AbstractVector` with `length(x) >= length(b)`.

See [`skeworthogonalize`](@ref) if `w` does not need to be normalized.

For more information on possible skew-orthogonalization algorithms, see
[`SkewOrthogonalizer`](@ref) and its concrete subtypes
[`ClassicalSymplecticGramSchmidt`](@ref), [`ModifiedSymplecticGramSchmidt`](@ref),
[`ClassicalSymplecticGramSchmidt2`](@ref), [`ModifiedSymplecticGramSchmidt2`](@ref),
[`ClassicalSymplecticGramSchmidtIR`](@ref) and [`ModifiedSymplecticGramSchmidtIR`](@ref).
"""
skeworthonormalize, skeworthonormalize!!
