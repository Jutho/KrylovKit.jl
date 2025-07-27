# BlockLanczos
"""
    x₀ = Block{S}(vec)
    x₀ = Block(vec)

Structure for storing vectors in a block format. The type parameter `T` represents the type of vector elements,
while `S` represents the type of inner products between vectors. To create an instance of the `Block` type,
one can specify `S` explicitly and use `Block{S}(vec)`, or use `Block(vec)` directly, in which case an
inner product is performed to infer `S`.
"""
struct Block{T,S<:Number}
    vec::Vector{T}
    function Block{S}(vec::Vector{T}) where {T,S<:Number}
        @assert length(vec) > 0
        return new{T,S}(vec)
    end
end
function Block(vec::Vector{T}) where {T}
    @assert length(vec) > 0
    S = typeof(inner(vec[1], vec[1]))
    return Block{S}(vec)
end # A convenient constructor for users
Base.length(b::Block) = length(b.vec)
Base.getindex(b::Block, i::Int) = b.vec[i]
function Base.getindex(b::Block{T,S}, idxs::AbstractVector{Int}) where {T,S}
    return Block{S}([b.vec[i] for i in idxs])
end
Base.setindex!(b::Block{T}, v::T, i::Int) where {T} = (b.vec[i] = v)
function Base.setindex!(b₁::Block{T}, b₂::Block{T},
                        idxs::AbstractVector{Int}) where {T}
    return (b₁.vec[idxs] = b₂.vec;
            b₁)
end
LinearAlgebra.norm(b::Block) = norm(b.vec)
function apply(f, block::Block{T,S}) where {T,S}
    return Block{S}([apply(f, block[i]) for i in 1:length(block)])
end
function block_inner(B₁::Block{T,S}, B₂::Block{T,S}) where {T,S}
    M = Matrix{S}(undef, length(B₁.vec), length(B₂.vec))
    @inbounds for j in 1:length(B₂)
        yj = B₂[j]
        for i in 1:length(B₁)
            M[i, j] = inner(B₁[i], yj)
        end
    end
    return M
end
function Base.push!(V::OrthonormalBasis{T}, b::Block{T}) where {T}
    for i in 1:length(b)
        push!(V, b[i])
    end
    return V
end
function Base.copy(b::Block)
    return Block{typeof(b).parameters[2]}(scale.(b.vec, 1))
end
Base.iterate(b::Block) = iterate(b.vec)
Base.iterate(b::Block, state) = iterate(b.vec, state)

"""
    mutable struct BlockLanczosFactorization{T,S<:Number,SR<:Real} <: KrylovFactorization{T,S,SR}

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
mutable struct BlockLanczosFactorization{T,S<:Number,SR<:Real} <: KrylovFactorization{T,S}
    k::Int
    V::OrthonormalBasis{T}      # BlockLanczos Basis
    H::Matrix{S}                # block tridiagonal matrix, and S is the matrix element type
    R::Block{T,S}               # residual block
    R_size::Int                 # size of the residual block
    norm_R::SR                  # norm of the residual block
end
Base.length(fact::BlockLanczosFactorization) = fact.k
normres(fact::BlockLanczosFactorization) = fact.norm_R
basis(fact::BlockLanczosFactorization) = fact.V
residual(fact::BlockLanczosFactorization) = fact.R[1:(fact.R_size)]

"""
    struct BlockLanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    BlockLanczosIterator(f, x₀, maxdim, qr_tol, [orth::Orthogonalizer = KrylovDefaults.orth])

Iterator that takes a linear map `f::F` (supposed to be real symmetric or complex hermitian)
and an initial block `x₀::Block{T,S}` and generates an expanding `BlockLanczosFactorization` thereof. In
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
struct BlockLanczosIterator{F,T,S,O<:Orthogonalizer,S2<:Real} <: KrylovIterator{F,T}
    operator::F
    x₀::Block{T,S}
    maxdim::Int
    orth::O
    qr_tol::S2
    function BlockLanczosIterator{F,T,S,O,S2}(operator::F,
                                              x₀::Block{T,S},
                                              maxdim::Int,
                                              orth::O,
                                              qr_tol::Real) where {F,T,S,O<:Orthogonalizer,
                                                                   S2}
        return new{F,T,S,O,S2}(operator, x₀, maxdim, orth, qr_tol)
    end
end
function BlockLanczosIterator(operator::F,
                              x₀::Block{T,S},
                              maxdim::Int,
                              qr_tol::Real,
                              orth::O=ModifiedGramSchmidt2()) where {F,T,S,
                                                                     O<:Orthogonalizer}
    norm(x₀) < qr_tol && @error "initial vector should not have norm zero"
    orth != ModifiedGramSchmidt2() &&
        @error "BlockLanczosIterator only supports ModifiedGramSchmidt2 orthogonalizer"
    return BlockLanczosIterator{F,T,S,O}(operator, x₀, maxdim, orth, qr_tol)
end

function initialize(iter::BlockLanczosIterator{F,T,S};
                    verbosity::Int=KrylovDefaults.verbosity[]) where {F,T,S}
    X₀ = iter.x₀
    maxdim = iter.maxdim
    A = iter.operator
    BTD = zeros(S, maxdim, maxdim)

    # Orthogonalization of the initial block
    X₁ = copy(X₀)
    _, good_idx = block_qr!(X₁, iter.qr_tol)
    X₁ = X₁[good_idx]
    V = OrthonormalBasis(X₁.vec)
    bs = length(X₁) # block size of the first block

    AX₁ = apply(A, X₁)
    M₁ = block_inner(X₁, AX₁)
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
    return BlockLanczosFactorization(bs,
                                     V,
                                     BTD,
                                     AX₁,
                                     bs,
                                     norm_R)
end

function expand!(iter::BlockLanczosIterator{F,T,S},
                 state::BlockLanczosFactorization{T,S,SR};
                 verbosity::Int=KrylovDefaults.verbosity[]) where {F,T,S,SR}
    k = state.k
    R = state.R[1:(state.R_size)]
    bs = length(R)
    V = state.V

    # Calculate the new basis and B
    B, good_idx = block_qr!(R, iter.qr_tol)
    bs_next = length(good_idx)
    push!(V, R[good_idx])
    state.H[(k + 1):(k + bs_next), (k - bs + 1):k] = view(B, 1:bs_next, 1:bs)
    state.H[(k - bs + 1):k, (k + 1):(k + bs_next)] = view(B, 1:bs_next, 1:bs)'

    # Calculate the new residual and orthogonalize the new basis
    Rnext, Mnext = blocklanczosrecurrence(iter.operator, V, B, iter.orth)
    verbosity >= WARN_LEVEL && warn_nonhermitian(Mnext)

    state.H[(k + 1):(k + bs_next), (k + 1):(k + bs_next)] = view(Mnext, 1:bs_next,
                                                                 1:bs_next)
    state.R.vec[1:bs_next] .= Rnext.vec
    state.norm_R = norm(Rnext)
    state.k += bs_next
    state.R_size = bs_next

    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos expansion to dimension $(state.k): subspace normres = $(normres2string(state.norm_R))"
    end
end

function blocklanczosrecurrence(operator, V::OrthonormalBasis, B::AbstractMatrix,
                                orth::ModifiedGramSchmidt2)
    # Apply the operator and calculate the M. Get Xnext and Mnext.
    bs, bs_prev = size(B)
    S = eltype(B)
    k = length(V)
    X = Block{S}(V[(k - bs + 1):k])
    AX = apply(operator, X)
    M = block_inner(X, AX)
    # Calculate the new residual in AX.
    Xprev = Block{S}(V[(k - bs_prev - bs + 1):(k - bs)])
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

"""
    block_reorthogonalize!(R::Block{T,S}, V::OrthonormalBasis{T}) where {T,S}

This function orthogonalizes the vectors in `R` with respect to the previously orthonormalized set `V` by using the modified Gram-Schmidt process.
Specifically, it modifies each vector `R[i]` by projecting out its components along the directions spanned by `V`, i.e.,

```
    R[i] = R[i] - sum(j=1:length(V)) <R[i], V[j]> V[j]
```

Here,`⟨·,·⟩` denotes the inner product. The function assumes that `V` is already orthonormal.
"""
function block_reorthogonalize!(R::Block{T,S},
                                V::OrthonormalBasis{T}) where {T,S}
    for i in 1:length(R)
        for q in V
            R[i], _ = orthogonalize!!(R[i], q, ModifiedGramSchmidt())
        end
    end
    return R
end

function warn_nonhermitian(M::AbstractMatrix)
    if norm(M - M') > eps(real(eltype(M)))^(2 / 5)
        @warn "ignoring the antihermitian part of the block triangular matrix: operator might not be hermitian?" M
    end
end

"""
    block_qr!(block::Block{T,S}, tol::Real) where {T,S}

This function performs a QR factorization of a block of abstract vectors using the modified Gram-Schmidt process.

```
    [v₁,..,vₚ] -> [u₁,..,uᵣ] * R
```

It takes as input a block of abstract vectors and a tolerance parameter, which is used to determine whether a vector is considered numerically zero.
The operation is performed in-place, transforming the input block into a block of orthonormal vectors.

The function returns a matrix of size `(r, p)` and a vector of indices goodidx. Here, `p` denotes the number of input vectors,
and `r` is the numerical rank of the input block. The matrix represents the upper-triangular factor of the QR decomposition,
restricted to the `r` linearly independent components. The vector `goodidx` contains the indices of the non-zero
(i.e., numerically independent) vectors in the orthonormalized block.
"""
function block_qr!(block::Block{T,S}, tol::Real) where {T,S}
    n = length(block)
    rank_shrink = false
    idx = trues(n)
    R = zeros(S, n, n)
    @inbounds for j in 1:n
        for i in 1:(j - 1)
            R[i, j] = inner(block[i], block[j])
            block[j] = add!!(block[j], block[i], -R[i, j])
        end
        β = norm(block[j])
        if !(β ≤ tol)
            R[j, j] = β
            block[j] = scale!!(block[j], 1 / β)
        else
            block[j] = zerovector!!(block[j])
            rank_shrink = true
            idx[j] = false
        end
    end
    good_idx = findall(idx)
    return R[good_idx, :], good_idx
end
