# BlockLanczos
"""
    struct BlockVec{T,S<:Number}

Structure for storing vectors in a block format. The type parameter `T` represents the type of vector elements,
while `S` represents the type of inner products between vectors.
"""
struct BlockVec{T,S<:Number}
    vec::Vector{T}
    function BlockVec{S}(vec::Vector{T}) where {T,S<:Number}
        return new{T,S}(vec)
    end
end
Base.length(b::BlockVec) = length(b.vec)
Base.getindex(b::BlockVec, i::Int) = b.vec[i]
function Base.getindex(b::BlockVec{T,S}, idxs::AbstractVector{Int}) where {T,S}
    return BlockVec{S}([b.vec[i] for i in idxs])
end
Base.setindex!(b::BlockVec{T}, v::T, i::Int) where {T} = (b.vec[i] = v)
function Base.setindex!(b₁::BlockVec{T}, b₂::BlockVec{T},
                        idxs::AbstractVector{Int}) where {T}
    return (b₁.vec[idxs] = b₂.vec;
            b₁)
end
LinearAlgebra.norm(b::BlockVec) = norm(b.vec)
function apply(f, block::BlockVec{T,S}) where {T,S}
    return BlockVec{S}([apply(f, block[i]) for i in 1:length(block)])
end
function Base.push!(V::OrthonormalBasis{T}, b::BlockVec{T}) where {T}
    for i in 1:length(b)
        push!(V, b[i])
    end
    return V
end
function Base.copy(b::BlockVec)
    return BlockVec{typeof(b).parameters[2]}(scale.(b.vec, 1))
end
Base.iterate(b::BlockVec) = iterate(b.vec)
Base.iterate(b::BlockVec, state) = iterate(b.vec, state)

"""
    mutable struct BlockLanczosFactorization{T,S<:Number,SR<:Real} <: BlockKrylovFactorization{T,S,SR}

Structure to store a BlockLanczos factorization of a real symmetric or complex hermitian linear
map `A` of the form

```julia
A * V = V * B + r * b'
```

For a given BlockLanczos factorization `fact`, length `k = length(fact)` and basis `V = basis(fact)` are
like [`LanczosFactorization`](@ref). The block tridiagonal matrix `T` is preallocated in `BlockLanczosFactorization`
and is of type `Hermitian{S<:Number}` with `size(T) == (krylovdim + bs₀, krylovdim + bs₀)` where `bs₀` is the size of the initial block
and `krylovdim` is the maximum dimension of the Krylov subspace. The residuals `r` is of type `Vector{T}`.
One can also query [`normres(fact)`](@ref) to obtain `norm(r)`, the norm of the residual. The matrix
`b` takes the default value ``[0;I]``, i.e. the matrix of size `(k,bs)` and an unit matrix in the last
`bs` rows and all zeros in the other rows. `bs` is the size of the last block. One can query [`r_size(fact)`] to obtain
size of the last block and the residuals.

`BlockLanczosFactorization` is mutable because it can [`expand!`](@ref). But it does not support `shrink!`
because it is implemented in its `eigsolve`.
See also [`BlockLanczosIterator`](@ref) for an iterator that constructs a progressively expanding
BlockLanczos factorizations of a given linear map and a starting vector.
"""
mutable struct BlockLanczosFactorization{T,S<:Number,SR<:Real} <:
               KrylovFactorization{T,S}
    k::Int
    const V::OrthonormalBasis{T}      # BlockLanczos Basis
    const T::AbstractMatrix{S}      # block tridiagonal matrix, and S is the matrix element type
    const r::BlockVec{T,S}            # residual block
    r_size::Int # size of the residual block
    norm_r::SR  # norm of the residual block
end
Base.length(fact::BlockLanczosFactorization) = fact.k
normres(fact::BlockLanczosFactorization) = fact.norm_r
basis(fact::BlockLanczosFactorization) = fact.V
residual(fact::BlockLanczosFactorization) = fact.r[1:(fact.r_size)]

"""
    struct BlockLanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    BlockLanczosIterator(f, x₀, maxdim, qr_tol, [orth::Orthogonalizer = KrylovDefaults.orth])

Iterator that takes a linear map `f::F` (supposed to be real symmetric or complex hermitian)
and an initial block `x₀::BlockVec{T,S}` and generates an expanding `BlockLanczosFactorization` thereof. In
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
`qr_tol` is the tolerance used in [`abstract_qr!`](@ref) to resolve the rank of a block of vectors.

When iterating over an instance of `BlockLanczosIterator`, the values being generated are
instances of [`BlockLanczosFactorization`](@ref). 

The internal state of `BlockLanczosIterator` is the same as the return value, i.e. the
corresponding `BlockLanczosFactorization`.

Here, [`initialize(::KrylovIterator)`](@ref) produces the first Krylov factorization,
and [`expand!(iter::KrylovIterator, fact::KrylovFactorization)`](@ref) expands the
factorization in place.
"""
struct BlockLanczosIterator{F,T,S,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::BlockVec{T,S}
    maxdim::Int
    orth::O
    qr_tol::Real
    function BlockLanczosIterator{F,T,S,O}(operator::F,
                                           x₀::BlockVec{T,S},
                                           maxdim::Int,
                                           orth::O,
                                           qr_tol::Real) where {F,T,S,O<:Orthogonalizer}
        return new{F,T,S,O}(operator, x₀, maxdim, orth, qr_tol)
    end
end
function BlockLanczosIterator(operator::F,
                              x₀::BlockVec{T,S},
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
    bs = length(X₀) # block size now
    A = iter.operator
    BTD = zeros(S, maxdim, maxdim)

    # Orthogonalization of the initial block
    X₁ = copy(X₀)
    abstract_qr!(X₁, iter.qr_tol)
    V = OrthonormalBasis(X₁.vec)

    AX₁ = apply(A, X₁)
    M₁ = block_inner(X₁, AX₁)
    BTD[1:bs, 1:bs] .= M₁
    verbosity >= WARN_LEVEL && warn_nonhermitian(M₁)

    # Get the first residual
    for j in 1:length(X₁)
        for i in 1:length(X₁)
            AX₁[j] = add!!(AX₁[j], X₁[i], -M₁[i, j])
        end
    end
    norm_r = norm(AX₁)
    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos initiation at dimension $bs: subspace normres = $(normres2string(norm_r))"
    end
    return BlockLanczosFactorization(bs,
                                     V,
                                     BTD,
                                     AX₁,
                                     bs,
                                     norm_r)
end

function expand!(iter::BlockLanczosIterator{F,T,S},
                 state::BlockLanczosFactorization{T,S,SR};
                 verbosity::Int=KrylovDefaults.verbosity[]) where {F,T,S,SR}
    k = state.k
    rₖ = state.r[1:(state.r_size)]
    bs_now = length(rₖ)
    V = state.V

    # Calculate the new basis and Bₖ
    Bₖ, good_idx = abstract_qr!(rₖ, iter.qr_tol)
    bs_next = length(good_idx)
    push!(V, rₖ[good_idx])
    state.T[(k + 1):(k + bs_next), (k - bs_now + 1):k] .= Bₖ
    state.T[(k - bs_now + 1):k, (k + 1):(k + bs_next)] .= Bₖ'

    # Calculate the new residual and orthogonalize the new basis
    rₖnext, Mnext = blocklanczosrecurrence(iter.operator, V, Bₖ, iter.orth)
    verbosity >= WARN_LEVEL && warn_nonhermitian(Mnext)

    state.T[(k + 1):(k + bs_next), (k + 1):(k + bs_next)] .= Mnext
    state.r.vec[1:bs_next] .= rₖnext.vec
    state.norm_r = norm(rₖnext)
    state.k += bs_next
    state.r_size = bs_next

    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos expansion to dimension $(state.k): subspace normres = $(normres2string(state.norm_r))"
    end
end

function blocklanczosrecurrence(operator, V::OrthonormalBasis, Bₖ::AbstractMatrix,
                                orth::ModifiedGramSchmidt2)
    # Apply the operator and calculate the M. Get Xnext and Mnext.
    bs, bs_last = size(Bₖ)
    S = eltype(Bₖ)
    k = length(V)
    X = BlockVec{S}(V[(k - bs + 1):k])
    AX = apply(operator, X)
    M = block_inner(X, AX)
    # Calculate the new residual. Get Rnext
    Xlast = BlockVec{S}(V[(k - bs_last - bs + 1):(k - bs)])
    rₖnext = compute_residual!(AX, X, M, Xlast, Bₖ')
    block_reorthogonalize!(rₖnext, V)
    return rₖnext, M
end

"""
    compute_residual!(AX::BlockVec{T,S}, X::BlockVec{T,S},
                           M::AbstractMatrix,
                           X_prev::BlockVec{T,S}, B_prev::AbstractMatrix) where {T,S}

Computes the residual block and stores the result in `AX`.

This function orthogonalizes `AX` against the two most recent basis blocks, `X` and `X_prev`.  
Here, `AX` represents the image of the current block `X` under the action of the linear operator `A`.  
The matrix `M` contains the inner products between `X` and `AX`, i.e., the projection of `AX` onto `X`.  
Similarly, `B_prev` represents the projection of `AX` onto `X_prev`.

The residual is computed as:

```
    AX ← AX - X * M - X_prev * B_prev
```

After this operation, `AX` is orthogonal (in the block inner product sense) to both `X` and `X_prev`.

"""
function compute_residual!(AX::BlockVec{T,S}, X::BlockVec{T,S},
                           M::AbstractMatrix,
                           X_prev::BlockVec{T,S}, B_prev::AbstractMatrix) where {T,S}
    @inbounds for j in 1:length(X)
        for i in 1:length(X)
            AX[j] = add!!(AX[j], X[i], -M[i, j])
        end
        for i in 1:length(X_prev)
            AX[j] = add!!(AX[j], X_prev[i], -B_prev[i, j])
        end
    end
    return AX
end

"""
    block_reorthogonalize!(basis::BlockVec{T,S}, basis_sofar::OrthonormalBasis{T}) where {T,S}

This function orthogonalizes the vectors in `basis` with respect to the previously orthonormalized set `basis_sofar` by using the modified Gram-Schmidt process.
Specifically, it modifies each vector `basis[i]` by projecting out its components along the directions spanned by `basis_sofar`, i.e.,

```
    basis[i] = basis[i] - sum(j=1:length(basis_sofar)) <basis[i], basis_sofar[j]> basis_sofar[j]
```

Here,`⟨·,·⟩` denotes the inner product. The function assumes that `basis_sofar` is already orthonormal.
"""
function block_reorthogonalize!(basis::BlockVec{T,S},
                                basis_sofar::OrthonormalBasis{T}) where {T,S}
    for i in 1:length(basis)
        for q in basis_sofar
            basis[i], _ = orthogonalize!!(basis[i], q, ModifiedGramSchmidt())
        end
    end
    return basis
end

function warn_nonhermitian(M::AbstractMatrix)
    if norm(M - M') > eps(real(eltype(M)))^(2 / 5)
        @warn "ignoring the antihermitian part of the block triangular matrix: operator might not be hermitian?" M
    end
end

"""
    abstract_qr!(block::BlockVec{T,S}, tol::Real) where {T,S}

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
function abstract_qr!(block::BlockVec{T,S}, tol::Real) where {T,S}
    n = length(block)
    rank_shrink = false
    idx = ones(Int64, n)
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
            idx[j] = 0
        end
    end
    good_idx = findall(idx .> 0)
    return R[good_idx, :], good_idx
end
