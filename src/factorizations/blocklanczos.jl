"""
    abstract type AbstractBlock

Supertype of the blocks of vectors that [`BlockLanczos`](@ref) works on: [`Block`](@ref), a list of
vectors, and [`MatrixBlock`](@ref), the columns of a single matrix-like object.
"""
abstract type AbstractBlock end

"""
    x₀ = Block{S}(vec)
    x₀ = Block(vec)

Structure for storing vectors in a block format. The type parameter `T` represents the type of vector elements,
while `S` represents the type of inner products between vectors. To create an instance of the `Block` type,
one can specify `S` explicitly and use `Block{S}(vec)`, or use `Block(vec)` directly, in which case an
inner product is performed to infer `S`.
"""
struct Block{T} <: AbstractBlock
    vec::Vector{T}
    function Block{T}(vec::Vector{T}) where {T}
        length(vec) > 0 || throw(ArgumentError("blocklength must be >(0)"))
        return new{T}(vec)
    end
end
Block(vec::Vector{T}) where {T} = Block{T}(vec)

Base.length(b::Block) = length(b.vec)

Base.@propagate_inbounds Base.getindex(b::Block, i::Int) = b.vec[i]
Base.@propagate_inbounds function Base.getindex(b::Block, idxs::AbstractVector{<:Integer})
    return Block(b.vec[idxs])
end

Base.@propagate_inbounds function Base.setindex!(b::Block, v, i::Int)
    b.vec[i] = v
    return b
end
Base.@propagate_inbounds function Base.setindex!(
        b₁::Block, b₂::Block, idxs::AbstractVector{Int}
    )
    b₁.vec[idxs] = b₂.vec
    return b₁
end

LinearAlgebra.norm(b::Block) = norm(b.vec)

apply(f, block::Block) = Block(map(Base.Fix1(apply, f), block.vec))

apply_scalartype(f, x₀::Block, as::Number...) = apply_scalartype(f, first(x₀), as...)

function block_inner(B₁::Block{T}, B₂::Block{T}) where {T}
    m₁₁ = inner(B₁[1], B₂[1])
    M = Matrix{typeof(m₁₁)}(undef, length(B₁), length(B₂))
    @inbounds M[1, 1] = m₁₁
    @inbounds for j in axes(M, 2), i in axes(M, 1)
        i == j == 1 && continue
        M[i, j] = inner(B₁[i], B₂[j])
    end
    return M
end

function Base.push!(V::OrthonormalBasis{T}, b::Block{T}) where {T}
    for i in 1:length(b)
        push!(V, b[i])
    end
    return V
end

Base.copy(b::Block{T}) where {T} = Block{T}(scale.(b.vec, One()))

Base.iterate(b::Block) = iterate(b.vec)
Base.iterate(b::Block, state) = iterate(b.vec, state)

"""
    x₀ = MatrixBlock(X)

A block whose vectors are the columns of a single matrix-like object `X`, such as a `Matrix`.
[`BlockLanczos`](@ref) then applies the linear map to the whole block at once, as `A * X` if `A`
is an `AbstractMatrix` and as `A(X)` otherwise, and computes inner products, linear combinations
and orthonormalizations of the block with matrix products and dense factorizations. The
eigenvectors and residuals that `eigsolve` returns are again `MatrixBlock`s, one column per
eigenvalue.

Every `AbstractMatrix` is supported (the orthonormalization uses `qr!` and `lmul!` with its `Q`).
Another matrix-like type `M`, whose columns are vectors with the Euclidean inner product, needs
methods for `MatrixBlock{M}` of `length` (the number of columns), `LinearAlgebra.norm`, `copy`,
[`block_inner`](@ref), [`block_mul!`](@ref), [`block_mul`](@ref) and [`block_orth!`](@ref).
"""
struct MatrixBlock{M} <: AbstractBlock
    data::M
end

Base.length(b::MatrixBlock{<:AbstractMatrix}) = size(b.data, 2)
LinearAlgebra.norm(b::MatrixBlock) = norm(b.data)
Base.copy(b::MatrixBlock) = MatrixBlock(copy(b.data))

apply(A::AbstractMatrix, b::MatrixBlock) = MatrixBlock(A * b.data)
apply(f, b::MatrixBlock) = MatrixBlock(f(b.data))

function apply_scalartype(f, x₀::MatrixBlock)
    Tfx = Base.promote_op(apply, typeof(f), typeof(x₀))
    if Tfx !== Union{} && Tfx !== Any
        T = eltype(Base.promote_op(block_inner, typeof(x₀), Tfx))
        T <: Number && return T
    end
    return eltype(block_inner(x₀, apply(f, x₀)))
end

"""
    block_inner(X::MatrixBlock, Y::MatrixBlock) -> AbstractMatrix

The matrix of inner products `Xᴴ Y` of the columns of `X` and `Y`, as a `Matrix`.
"""
block_inner(X::MatrixBlock{<:AbstractMatrix}, Y::MatrixBlock{<:AbstractMatrix}) =
    convert(Matrix, X.data' * Y.data)

"""
    block_mul!(Y::MatrixBlock, X::MatrixBlock, C::AbstractMatrix, α, β) -> Y

`Y = β Y + α X C` in place, for a `Matrix` `C`.
"""
function block_mul!(Y::MatrixBlock{<:AbstractMatrix}, X::MatrixBlock{<:AbstractMatrix}, C::AbstractMatrix, α, β)
    mul!(Y.data, X.data, _blocklike(X.data, C), α, β)
    return Y
end

"""
    block_mul(X::MatrixBlock, C::AbstractMatrix) -> MatrixBlock

The new block `X C`, for a `Matrix` `C`.
"""
block_mul(X::MatrixBlock{<:AbstractMatrix}, C::AbstractMatrix) = MatrixBlock(X.data * _blocklike(X.data, C))

# the small matrix `C` with the storage of the block `X` (copied contiguously first: GPU copies need a dense source)
_blocklike(::Array, C::AbstractMatrix) = C
_blocklike(X::AbstractMatrix, C::AbstractMatrix) = copyto!(similar(X, eltype(C), size(C)), Matrix(C))

"""
    block_orth!(X::MatrixBlock, tol::Real) -> Q, B, is_drift

Orthonormal basis `Q` of the range of `X` with `X = Q B` up to the directions whose singular value
is below `tol`, which are dropped (`B` has as many rows as `Q` has columns). `X` may be
overwritten. `is_drift` is `true` if the smallest kept singular value is below `100 * tol`, so
that `Q` may have lost orthogonality to vectors that were projected out of `X` before.
"""
function block_orth!(X::MatrixBlock{<:AbstractMatrix}, tol::Real)
    A = X.data
    F = qr!(A)
    U, σ, W = svd!(convert(Matrix, F.R))
    r = count(>(tol), σ)
    Q = fill!(similar(A, size(A, 1), r), zero(eltype(A)))
    view(Q, 1:size(U, 1), :) .= _blocklike(A, view(U, :, 1:r))
    lmul!(F.Q, Q)
    B = Diagonal(view(σ, 1:r)) * view(W, :, 1:r)'
    return MatrixBlock(Q), B, r > 0 && σ[r] < 100 * tol
end

"""
    MatrixBlockBasis{M}

Orthonormal basis of a [`BlockLanczosFactorization`](@ref) started from a [`MatrixBlock`](@ref):
the list of its blocks, whose columns together are orthonormal.
"""
struct MatrixBlockBasis{M}
    blocks::Vector{MatrixBlock{M}}
end
Base.length(V::MatrixBlockBasis) = sum(length, V.blocks; init = 0)
Base.push!(V::MatrixBlockBasis{M}, b::MatrixBlock{M}) where {M} = (push!(V.blocks, b); V)

# `V U`, as one block; `U` has a row per basis vector
function basis_mul(V::MatrixBlockBasis, U::AbstractMatrix)
    b₁ = first(V.blocks)
    i = length(b₁)
    Y = block_mul(b₁, view(U, 1:i, :))
    for b in Iterators.drop(V.blocks, 1)
        k = length(b)
        block_mul!(Y, b, view(U, (i + 1):(i + k), :), true, true)
        i += k
    end
    return Y
end

# R - V Vᴴ R in place, one modified Gram-Schmidt step per block of V
function block_reorthogonalize!(R::MatrixBlock, V::MatrixBlockBasis)
    for b in V.blocks
        block_mul!(R, b, block_inner(b, R), -1, true)
    end
    return R
end

"""
    mutable struct BlockLanczosFactorization{T,S<:Number,SR<:Real} <: KrylovFactorization{T,S}

Structure to store a BlockLanczos factorization of a real symmetric or complex hermitian linear
map `A` of the form

```julia
A * V = V * H + R * B'
```

For a given BlockLanczos factorization `fact`, length `k = length(fact)` and basis `V = basis(fact)` are
like [`LanczosFactorization`](@ref). The hermitian block tridiagonal matrix `H` is preallocated
in `BlockLanczosFactorization` and can reach a maximal size of `(krylovdim + bs₀, krylovdim + bs₀)`, where `bs₀` is the size of the initial block
and `krylovdim` is the maximum dimension of the Krylov subspace. A list of residual vectors is contained in `R` is of type `Vector{T}`.
One can also query [`normres(fact)`](@ref) to obtain `norm(R)`, which computes a total norm of all residual vectors combined. The matrix
`B` takes the default value ``[0; I]``, i.e. the matrix of size `(k,bs)` containing a unit matrix in the last
`bs` rows and all zeros in the other rows. `bs` is the size of the last block.

`BlockLanczosFactorization` is mutable because it can [`expand!`](@ref). But it does not support `shrink!`
because it is implemented in its `eigsolve`.
See also [`BlockLanczosIterator`](@ref) for an iterator that constructs a progressively expanding
BlockLanczos factorizations of a given linear map and a starting block.
"""
mutable struct BlockLanczosFactorization{T, S <: Number, SR <: Real, VB, RB <: AbstractBlock} <: KrylovFactorization{T, S}
    k::Int
    V::VB                       # BlockLanczos Basis: OrthonormalBasis{T}, or MatrixBlockBasis{T} for a MatrixBlock
    H::Matrix{S}                # block tridiagonal matrix, and S is the matrix element type
    R::RB                       # residual block
    R_size::Int                 # size of the residual block
    norm_R::SR                  # norm of the residual block
end
function BlockLanczosFactorization(
        k::Int, V::OrthonormalBasis{T}, H::Matrix{S}, R::Block{T}, R_size::Int, norm_R::SR
    ) where {T, S, SR}
    return BlockLanczosFactorization{T, S, SR, typeof(V), typeof(R)}(k, V, H, R, R_size, norm_R)
end
function BlockLanczosFactorization(
        k::Int, V::MatrixBlockBasis{T}, H::Matrix{S}, R::MatrixBlock{T}, R_size::Int, norm_R::SR
    ) where {T, S, SR}
    return BlockLanczosFactorization{T, S, SR, typeof(V), typeof(R)}(k, V, H, R, R_size, norm_R)
end
const MatrixBlockLanczosFactorization = BlockLanczosFactorization{T, S, SR, VB, <:MatrixBlock} where {T, S <: Number, SR <: Real, VB}
Base.length(fact::BlockLanczosFactorization) = fact.k
normres(fact::BlockLanczosFactorization) = fact.norm_R
basis(fact::BlockLanczosFactorization) = fact.V
residual(fact::BlockLanczosFactorization) = fact.R[1:(fact.R_size)]
residual(fact::MatrixBlockLanczosFactorization) = fact.R

"""
    struct BlockLanczosIterator{F,T,S<:Real,O<:Orthogonalizer} <: KrylovIterator{F,T}
    BlockLanczosIterator(f, x₀, maxdim, qr_tol, [orth::Orthogonalizer = KrylovDefaults.orth])

Iterator that takes a linear map `f::F` (supposed to be real symmetric or complex hermitian)
and an initial block `x₀::Block{T}` and generates an expanding `BlockLanczosFactorization` thereof. In
particular, `BlockLanczosIterator` uses the
BlockLanczos iteration(see: *Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations* (4th ed., pp. 566–569))
scheme to build a successively expanding BlockLanczos factorization. While `f` cannot be tested to be symmetric or
hermitian directly when the linear map is encoded as a general callable object or function, with `block_inner(X, f.(X))`,
it is tested whether `norm(M-M')` is sufficiently small to be neglected.

The argument `f` can be a matrix, or a function accepting a single argument `v`, so that
`f(v)` implements the action of the linear map on the vector `v`.

The optional argument `orth` specifies which [`Orthogonalizer`](@ref) to be used. The
default value in [`KrylovDefaults`](@ref) is to use [`ModifiedGramSchmidt2`](@ref), which
uses reorthogonalization steps in every iteration.
Now our orthogonalizer is only ModifiedGramSchmidt2. So we don't need to provide "keepvecs" because we have to reverse all krylove vectors.
Dimension of Krylov subspace in BlockLanczosIterator is usually much bigger than lanczos and its Default value is 100.
`qr_tol` is the tolerance used in [`block_qr!`](@ref) to resolve the rank of a block of vectors.

When iterating over an instance of `BlockLanczosIterator`, the values being generated are
instances of [`BlockLanczosFactorization`](@ref). 

The internal state of `BlockLanczosIterator` is the same as the return value, i.e. the
corresponding `BlockLanczosFactorization`.

Here, [`initialize(::KrylovIterator)`](@ref) produces the first Krylov factorization,
and [`expand!(iter::KrylovIterator, fact::KrylovFactorization)`](@ref) expands the
factorization in place.
"""
struct BlockLanczosIterator{F, T, O <: Orthogonalizer, S <: Real, B <: AbstractBlock} <: KrylovIterator{F, T}
    operator::F
    x₀::B
    maxdim::Int
    orth::O
    qr_tol::S
    function BlockLanczosIterator{F, T, O, S}(
            operator::F, x₀::B, maxdim::Int, orth::O, qr_tol::S
        ) where {F, T, O <: Orthogonalizer, S <: Real, B <: Union{Block{T}, MatrixBlock{T}}}
        return new{F, T, O, S, B}(operator, x₀, maxdim, orth, qr_tol)
    end
end
function BlockLanczosIterator(
        operator::F, x₀::Union{Block{T}, MatrixBlock{T}}, maxdim::Int,
        orth::O = KrylovDefaults.orth, qr_tol::Real = KrylovDefaults.tol[]
    ) where {
        F, T,
        O <: Orthogonalizer,
    }
    norm(x₀) < qr_tol && @error "initial vector should not have norm zero"
    orth != ModifiedGramSchmidt2() &&
        @error "BlockLanczosIterator only supports ModifiedGramSchmidt2 orthogonalizer"
    return BlockLanczosIterator{F, T, O, typeof(qr_tol)}(operator, x₀, maxdim, orth, qr_tol)
end
const MatrixBlockLanczosIterator = BlockLanczosIterator{F, T, O, S, <:MatrixBlock} where {F, T, O <: Orthogonalizer, S <: Real}

function initialize(
        iter::BlockLanczosIterator;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    X₀ = iter.x₀
    maxdim = iter.maxdim
    A = iter.operator

    # Some overhead to determine the proper types for the rest of the computation
    β₀ = norm(X₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    x₀ = X₀[1]
    Ax₀ = apply(A, x₀) # one extra application of the operator: can be avoided but is it worth it?
    α = inner(x₀, Ax₀) / β₀ # this value has no meaning, we only care about its type
    X₁ = Block(map(Base.Fix2(scale, one(α)), X₀.vec))

    # Orthogonalization of the initial block
    _, good_idx, _ = block_qr!(X₁, iter.qr_tol)
    X₁ = X₁[good_idx]
    V = OrthonormalBasis(X₁.vec)
    bs = length(X₁) # block size of the first block

    AX₁ = apply(A, X₁)
    M₁ = block_inner(X₁, AX₁)
    BTD = zeros(eltype(M₁), maxdim, maxdim)
    BTD[1:bs, 1:bs] = view(M₁, 1:bs, 1:bs)
    verbosity >= WARN_LEVEL && warn_nonhermitian(M₁)

    # Get the first residual
    for j in 1:length(X₁)
        for i in 1:length(X₁)
            AX₁[j] = add!!(AX₁[j], X₁[i], -M₁[i, j])
        end
    end
    norm_R = norm(AX₁)
    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos initiation at dimension $bs: subspace normres = $(normres2string(norm_R))"
    end
    return BlockLanczosFactorization(bs, V, BTD, AX₁, bs, norm_R)
end

function expand!(
        iter::BlockLanczosIterator, state::BlockLanczosFactorization;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    k = state.k
    R = state.R[1:(state.R_size)]
    bs = length(R)
    V = state.V
    Rcopy = copy(R)

    # Calculate the new basis and B
    B, good_idx, is_drift = block_qr!(R, iter.qr_tol)
    if is_drift # Prevent column subspace of R from drifting caused by an excessively small β in block_qr!
        block_reorthogonalize!(R, V)
        _, good_idx, is_drift = block_qr!(R, iter.qr_tol)
        B = block_inner(R[good_idx], Rcopy) # Make sure R = XB
    end

    bs_next = length(good_idx)
    push!(V, R[good_idx])
    state.H[(k + 1):(k + bs_next), (k - bs + 1):k] = view(B, 1:bs_next, 1:bs)
    state.H[(k - bs + 1):k, (k + 1):(k + bs_next)] = view(B, 1:bs_next, 1:bs)'

    # Calculate the new residual and orthogonalize the new basis
    Rnext, Mnext = block_lanczosrecurrence(iter.operator, V, B, iter.orth)
    verbosity >= WARN_LEVEL && warn_nonhermitian(Mnext)

    state.H[(k + 1):(k + bs_next), (k + 1):(k + bs_next)] = view(
        Mnext, 1:bs_next,
        1:bs_next
    )
    state.R.vec[1:bs_next] = Rnext.vec
    state.norm_R = norm(Rnext)
    state.k += bs_next
    state.R_size = bs_next

    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos expansion to dimension $(state.k): subspace normres = $(normres2string(state.norm_R))"
    end
    return state
end

function block_lanczosrecurrence(
        operator, V::OrthonormalBasis, B::AbstractMatrix, orth::ModifiedGramSchmidt2
    )
    # Apply the operator and calculate the M. Get Xnext and Mnext.
    bs, bs_prev = size(B)
    k = length(V)
    X = Block(V[(k - bs + 1):k])
    AX = apply(operator, X)
    M = block_inner(X, AX)
    # Calculate the new residual in AX.
    Xprev = Block(V[(k - bs_prev - bs + 1):(k - bs)])
    @inbounds for j in 1:length(X)
        for i in 1:length(X)
            AX[j] = add!!(AX[j], X[i], -M[i, j])
        end
        for i in 1:length(Xprev)
            AX[j] = add!!(AX[j], Xprev[i], -conj(B[j, i]))
        end
    end
    block_reorthogonalize!(AX, V)
    return AX, M
end

# A MatrixBlock: the same recurrence with one application of the operator and matrix products per block
function initialize(
        iter::MatrixBlockLanczosIterator;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    iszero(norm(iter.x₀)) && throw(ArgumentError("initial vector should not have norm zero"))
    X₁, = block_orth!(copy(iter.x₀), iter.qr_tol)
    AX₁ = apply(iter.operator, X₁)
    M₁ = block_inner(X₁, AX₁)
    S = eltype(M₁)
    scalartype(X₁.data) == S || (X₁ = MatrixBlock(scale(X₁.data, one(S))))
    bs = length(X₁)
    BTD = zeros(S, iter.maxdim, iter.maxdim)
    BTD[1:bs, 1:bs] = M₁
    verbosity >= WARN_LEVEL && warn_nonhermitian(M₁)

    R = block_mul!(AX₁, X₁, M₁, -1, true)
    norm_R = norm(R)
    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos initiation at dimension $bs: subspace normres = $(normres2string(norm_R))"
    end
    return BlockLanczosFactorization(bs, MatrixBlockBasis([X₁]), BTD, R, bs, norm_R)
end

function expand!(
        iter::MatrixBlockLanczosIterator,
        state::MatrixBlockLanczosFactorization;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    k = state.k
    V = state.V
    bs = length(state.R)

    # the next block X and B with R = X B
    X, B, is_drift = block_orth!(state.R, iter.qr_tol)
    if is_drift # the directions with small singular values may have drifted back into the basis
        block_reorthogonalize!(X, V)
        X, B′, = block_orth!(X, iter.qr_tol)
        B = B′ * B
    end
    bs_next = length(X)
    Xprev = V.blocks[end]
    push!(V, X)
    state.H[(k + 1):(k + bs_next), (k - bs + 1):k] = B
    state.H[(k - bs + 1):k, (k + 1):(k + bs_next)] = B'

    # the residual of the next block, with full reorthogonalization
    R = apply(iter.operator, X)
    M = block_inner(X, R)
    verbosity >= WARN_LEVEL && warn_nonhermitian(M)
    block_mul!(R, X, M, -1, true)
    block_mul!(R, Xprev, B', -1, true)
    block_reorthogonalize!(R, V)

    state.H[(k + 1):(k + bs_next), (k + 1):(k + bs_next)] = M
    state.R = R
    state.norm_R = norm(R)
    state.k += bs_next
    state.R_size = bs_next

    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos expansion to dimension $(state.k): subspace normres = $(normres2string(state.norm_R))"
    end
    return state
end

"""
    block_reorthogonalize!(R::Block{T}, V::OrthonormalBasis{T}) where {T}

This function orthogonalizes the vectors in `R` with respect to the previously orthonormalized set `V` by using the modified Gram-Schmidt process.
Specifically, it modifies each vector `R[i]` by projecting out its components along the directions spanned by `V`, i.e.,

```
    R[i] = R[i] - sum(j=1:length(V)) <R[i], V[j]> V[j]
```

Here,`⟨·,·⟩` denotes the inner product. The function assumes that `V` is already orthonormal.
"""
function block_reorthogonalize!(R::Block{T}, V::OrthonormalBasis{T}) where {T}
    for i in 1:length(R)
        for q in V
            R[i], _ = orthogonalize!!(R[i], q, ModifiedGramSchmidt())
        end
    end
    return R
end

function warn_nonhermitian(M::AbstractMatrix)
    return if !isapprox(M, M'; atol = eps(real(eltype(M)))^(2 / 5))
        @warn "ignoring the antihermitian part of the block triangular matrix: operator might not be hermitian?" M
    end
end

"""
    block_qr!(block::Block{T,S}, tol::Real) where {T,S}

This function performs a QR factorization of a block of abstract vectors using the modified Gram-Schmidt process
with column pivoting, i.e. the remaining vector with the largest norm is processed first.

```
    [v₁,..,vₚ] -> [u₁,..,uᵣ] * R
```

It takes as input a block of abstract vectors and a tolerance parameter, which is used to determine whether a vector is considered numerically zero.
The operation is performed in-place, transforming the input block into a block of orthonormal vectors.

The function returns a matrix of size `(r, p)`, a vector of indices `goodidx` and a boolean flag `is_drift`. Here, `p` denotes the number of input vectors,
and `r` is the numerical rank of the input block. The matrix represents the upper-triangular factor of the QR decomposition, up to a column permutation,
restricted to the `r` linearly independent components. The vector `goodidx` contains the indices of the non-zero
(i.e., numerically independent) vectors in the orthonormalized block, in pivot order.
If a small value of β (the norm of a vector after first orthogonalization) is detected, the function will carry out an additional
reorthogonalization step to further ensure the input block vectors are orthonormalized.
In such cases, is_drift is set to true to indicate potential numerical instability.
"""
function block_qr!(block::Block, tol::Real)
    n = length(block)
    is_drift = false
    r₁₁ = inner(block[1], block[1])
    R = zeros(typeof(r₁₁), n, n)
    remaining = collect(1:n)
    good_idx = Int[]
    while !isempty(remaining)
        j = popat!(remaining, argmax([norm(block[l]) for l in remaining]))
        β = norm(block[j])
        if tol < β < 100 * tol # DGKS reorthogonalization
            is_drift = true
            for (s, i) in enumerate(good_idx)
                δ = inner(block[i], block[j])
                R[s, j] += δ
                block[j] = add!!(block[j], block[i], -δ)
            end
            β = norm(block[j])
        end
        if β < tol
            block[j] = zerovector!!(block[j])
            continue
        end
        push!(good_idx, j)
        s = length(good_idx)
        R[s, j] = β
        block[j] = scale!!(block[j], 1 / β)
        for l in remaining # MGS
            R[s, l] = inner(block[j], block[l])
            block[l] = add!!(block[l], block[j], -R[s, l])
        end
    end
    return R[1:length(good_idx), :], good_idx, is_drift
end
