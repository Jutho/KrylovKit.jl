# lanczos.jl
"""
    mutable struct LanczosFactorization{T,S<:Real} <: KrylovFactorization{T,S}

Structure to store a Lanczos factorization of a real symmetric or complex hermitian linear
map `A` of the form

```julia
A * V = V * B + r * b'
```

For a given Lanczos factorization `fact` of length `k = length(fact)`, the basis `V` is
obtained via [`basis(fact)`](@ref basis) and is an instance of [`OrthonormalBasis{T}`](@ref
Basis), with also `length(V) == k` and where `T` denotes the type of vector like objects
used in the problem. The Rayleigh quotient `B` is obtained as
[`rayleighquotient(fact)`](@ref) and is of type `SymTridiagonal{S<:Real}` with `size(B) ==
(k,k)`. The residual `r` is obtained as [`residual(fact)`](@ref) and is of type `T`. One can
also query [`normres(fact)`](@ref) to obtain `norm(r)`, the norm of the residual. The vector
`b` has no dedicated name but can be obtained via [`rayleighextension(fact)`](@ref). It
takes the default value ``e_k``, i.e. the unit vector of all zeros and a one in the last
entry, which is represented using [`SimpleBasisVector`](@ref).

A Lanczos factorization `fact` can be destructured as `V, B, r, nr, b = fact` with
`nr = norm(r)`.

`LanczosFactorization` is mutable because it can [`expand!`](@ref) or [`shrink!`](@ref).
See also [`LanczosIterator`](@ref) for an iterator that constructs a progressively expanding
Lanczos factorizations of a given linear map and a starting vector. See
[`ArnoldiFactorization`](@ref) and [`ArnoldiIterator`](@ref) for a Krylov factorization that
works for general (non-symmetric) linear maps.
"""
mutable struct LanczosFactorization{T,S<:Real} <: KrylovFactorization{T,S}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    αs::Vector{S}
    βs::Vector{S}
    r::T
end

Base.length(F::LanczosFactorization) = F.k
Base.sizehint!(F::LanczosFactorization, n) = begin
    sizehint!(F.V, n)
    sizehint!(F.αs, n)
    sizehint!(F.βs, n)
    return F
end
Base.eltype(F::LanczosFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:LanczosFactorization{<:Any,S}}) where {S} = S

function basis(F::LanczosFactorization)
    return length(F.V) == F.k ? F.V :
           error("Not keeping vectors during Lanczos factorization")
end
rayleighquotient(F::LanczosFactorization) = SymTridiagonal(F.αs, F.βs)
residual(F::LanczosFactorization) = F.r
@inbounds normres(F::LanczosFactorization) = F.βs[F.k]
rayleighextension(F::LanczosFactorization) = SimpleBasisVector(F.k, F.k)

# Lanczos iteration for constructing the orthonormal basis of a Krylov subspace.
"""
    struct LanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    LanczosIterator(f, v₀, [orth::Orthogonalizer = KrylovDefaults.orth, keepvecs::Bool = true])

Iterator that takes a linear map `f::F` (supposed to be real symmetric or complex hermitian)
and an initial vector `v₀::T` and generates an expanding `LanczosFactorization` thereof. In
particular, `LanczosIterator` uses the
[Lanczos iteration](https://en.wikipedia.org/wiki/Lanczos_algorithm) scheme to build a
successively expanding Lanczos factorization. While `f` cannot be tested to be symmetric or
hermitian directly when the linear map is encoded as a general callable object or function,
it is tested whether the imaginary part of `inner(v, f(v))` is sufficiently small to be
neglected.

The argument `f` can be a matrix, or a function accepting a single argument `v`, so that
`f(v)` implements the action of the linear map on the vector `v`.

The optional argument `orth` specifies which [`Orthogonalizer`](@ref) to be used. The
default value in [`KrylovDefaults`](@ref) is to use [`ModifiedGramSchmidtIR`](@ref), which
possibly uses reorthogonalization steps. One can use to discard the old vectors that span
the Krylov subspace by setting the final argument `keepvecs` to `false`. This, however, is
only possible if an `orth` algorithm is used that does not rely on reorthogonalization, such
as `ClassicalGramSchmidt()` or `ModifiedGramSchmidt()`. In that case, the iterator strictly
uses the Lanczos three-term recurrence relation.

When iterating over an instance of `LanczosIterator`, the values being generated are
instances of [`LanczosFactorization`](@ref), which can be immediately destructured into a
[`basis`](@ref), [`rayleighquotient`](@ref), [`residual`](@ref), [`normres`](@ref) and
[`rayleighextension`](@ref), for example as

```julia
for (V, B, r, nr, b) in LanczosIterator(f, v₀)
    # do something
    nr < tol && break # a typical stopping criterion
end
```

Note, however, that if `keepvecs=false` in `LanczosIterator`, the basis `V` cannot be
extracted.

Since the iterator does not know the dimension of the underlying vector space of
objects of type `T`, it keeps expanding the Krylov subspace until the residual norm `nr`
falls below machine precision `eps(typeof(nr))`.

The internal state of `LanczosIterator` is the same as the return value, i.e. the
corresponding `LanczosFactorization`. However, as Julia's Base iteration interface (using
`Base.iterate`) requires that the state is not mutated, a `deepcopy` is produced upon every
next iteration step.

Instead, you can also mutate the `KrylovFactorization` in place, using the following
interface, e.g. for the same example above

```julia
iterator = LanczosIterator(f, v₀)
factorization = initialize(iterator)
while normres(factorization) > tol
    expand!(iterator, factorization)
    V, B, r, nr, b = factorization
    # do something
end
```

Here, [`initialize(::KrylovIterator)`](@ref) produces the first Krylov factorization of
length 1, and [`expand!(iter::KrylovIterator, fact::KrylovFactorization)`](@ref) expands the factorization in place. See also
factorization in place. See also [`initialize!(::KrylovIterator,
::KrylovFactorization)`](@ref) to initialize in an already existing factorization (most
information will be discarded) and [`shrink!(::KrylovFactorization, k)`](@ref) to shrink an
existing factorization down to length `k`.
"""
struct LanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::T
    orth::O
    keepvecs::Bool
    function LanczosIterator{F,T,O}(operator::F,
                                    x₀::T,
                                    orth::O,
                                    keepvecs::Bool) where {F,T,O<:Orthogonalizer}
        if !keepvecs && isa(orth, Reorthogonalizer)
            error("Cannot use reorthogonalization without keeping all Krylov vectors")
        end
        return new{F,T,O}(operator, x₀, orth, keepvecs)
    end
end
function LanczosIterator(operator::F,
                         x₀::T,
                         orth::O=KrylovDefaults.orth,
                         keepvecs::Bool=true) where {F,T,O<:Orthogonalizer}
    return LanczosIterator{F,T,O}(operator, x₀, orth, keepvecs)
end

Base.IteratorSize(::Type{<:LanczosIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:LanczosIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::LanczosIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::LanczosIterator, state::LanczosFactorization)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function warn_nonhermitian(α, β₁, β₂)
    n = hypot(α, β₁, β₂)
    if abs(imag(α)) / n > eps(one(n))^(2 / 5)
        @warn "ignoring imaginary component $(imag(α)) from total weight $n: operator might not be hermitian?" α β₁ β₂
    end
    return nothing
end

function initialize(iter::LanczosIterator; verbosity::Int=KrylovDefaults.verbosity[])
    # initialize without using eltype
    x₀ = iter.x₀
    β₀ = norm(x₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    Ax₀ = apply(iter.operator, x₀)
    α = inner(x₀, Ax₀) / (β₀ * β₀)
    T = typeof(α) # scalar type of the Rayleigh quotient
    # this line determines the vector type that we will henceforth use
    # vector scalar type can be different from `T`, e.g. for real inner products
    v = add!!(scale(Ax₀, zero(α)), x₀, 1 / β₀)
    if typeof(Ax₀) != typeof(v)
        r = add!!(zerovector(v), Ax₀, 1 / β₀)
    else
        r = scale!!(Ax₀, 1 / β₀)
    end
    βold = norm(r)
    r = add!!(r, v, -α) # should we use real(α) here?
    β = norm(r)
    # possibly reorthogonalize
    if iter.orth isa Union{ClassicalGramSchmidt2,ModifiedGramSchmidt2}
        dα = inner(v, r)
        α += dα
        r = add!!(r, v, -dα) # should we use real(dα) here?
        β = norm(r)
    elseif iter.orth isa Union{ClassicalGramSchmidtIR,ModifiedGramSchmidtIR}
        while eps(one(β)) < β < iter.orth.η * βold
            βold = β
            dα = inner(v, r)
            α += dα
            r = add!!(r, v, -dα) # should we use real(dα) here?
            β = norm(r)
        end
    end
    verbosity >= WARN_LEVEL && warn_nonhermitian(α, zero(β), β)
    V = OrthonormalBasis([v])
    αs = [real(α)]
    βs = [β]
    if verbosity > EACHITERATION_LEVEL
        @info "Lanczos initiation at dimension 1: subspace normres = $(normres2string(β))"
    end
    return LanczosFactorization(1, V, αs, βs, r)
end
function initialize!(iter::LanczosIterator, state::LanczosFactorization;
                     verbosity::Int=KrylovDefaults.verbosity[])
    x₀ = iter.x₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    αs = empty!(state.αs)
    βs = empty!(state.βs)

    V[1] = scale!!(V[1], x₀, 1 / norm(x₀))
    w = apply(iter.operator, V[1])
    r, α = orthogonalize!!(w, V[1], iter.orth)
    β = norm(r)
    verbosity >= WARN_LEVEL && warn_nonhermitian(α, zero(β), β)

    state.k = 1
    push!(αs, real(α))
    push!(βs, β)
    state.r = r
    if verbosity > EACHITERATION_LEVEL
        @info "Lanczos initiation at dimension 1: subspace normres = $(normres2string(β))"
    end
    return state
end
function expand!(iter::LanczosIterator, state::LanczosFactorization;
                 verbosity::Int=KrylovDefaults.verbosity[])
    βold = normres(state)
    V = state.V
    r = state.r
    V = push!(V, scale!!(r, 1 / βold))
    r, α, β = lanczosrecurrence(iter.operator, V, βold, iter.orth)
    verbosity >= WARN_LEVEL && warn_nonhermitian(α, βold, β)

    αs = push!(state.αs, real(α))
    βs = push!(state.βs, β)

    !iter.keepvecs && popfirst!(state.V) # remove oldest V if not keepvecs

    state.k += 1
    state.r = r
    if verbosity > EACHITERATION_LEVEL
        @info "Lanczos expansion to dimension $(state.k): subspace normres = $(normres2string(β))"
    end
    return state
end
function shrink!(state::LanczosFactorization, k; verbosity::Int=KrylovDefaults.verbosity[])
    length(state) == length(state.V) ||
        error("we cannot shrink LanczosFactorization without keeping Lanczos vectors")
    length(state) <= k && return state
    V = state.V
    while length(V) > k + 1
        pop!(V)
    end
    r = pop!(V)
    resize!(state.αs, k)
    resize!(state.βs, k)
    state.k = k
    β = normres(state)
    if verbosity > EACHITERATION_LEVEL
        @info "Lanczos reduction to dimension $k: subspace normres = $(normres2string(β))"
    end
    state.r = scale!!(r, β)
    return state
end

# Exploit hermiticity to "simplify" orthonormalization process:
# Lanczos three-term recurrence relation
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt)
    v = V[end]
    w = apply(operator, v)
    α = inner(v, w)
    w = add!!(w, V[end - 1], -β)
    w = add!!(w, v, -α)
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt)
    v = V[end]
    w = apply(operator, v)
    w = add!!(w, V[end - 1], -β)
    α = inner(v, w)
    w = add!!(w, v, -α)
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt2)
    v = V[end]
    w = apply(operator, v)
    α = inner(v, w)
    w = add!!(w, V[end - 1], -β)
    w = add!!(w, v, -α)

    w, s = orthogonalize!!(w, V, ClassicalGramSchmidt())
    α += s[end]
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt2)
    v = V[end]
    w = apply(operator, v)
    w = add!!(w, V[end - 1], -β)
    w, α = orthogonalize!!(w, v, ModifiedGramSchmidt())

    s = α
    for q in V
        w, s = orthogonalize!!(w, q, ModifiedGramSchmidt())
    end
    α += s
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidtIR)
    v = V[end]
    w = apply(operator, v)
    α = inner(v, w)
    w = add!!(w, V[end - 1], -β)
    w = add!!(w, v, -α)

    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        w, s = orthogonalize!!(w, V, ClassicalGramSchmidt())
        α += s[end]
        β = norm(w)
    end
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidtIR)
    v = V[end]
    w = apply(operator, v)
    w = add!!(w, V[end - 1], -β)

    w, α = orthogonalize!!(w, v, ModifiedGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        s = zero(α)
        for q in V
            w, s = orthogonalize!!(w, q, ModifiedGramSchmidt())
        end
        α += s
        β = norm(w)
    end
    return w, α, β
end

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
function initialize(x₀, size::Int)
    S = typeof(inner(x₀, x₀))
    x₀_vec = [randn!(similar(x₀)) for _ in 1:(size - 1)]
    pushfirst!(x₀_vec, x₀)
    return BlockVec{S}(x₀_vec)
end
function Base.push!(V::OrthonormalBasis{T}, b::BlockVec{T}) where {T}
    for i in 1:length(b)
        push!(V, b[i])
    end
    return V
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
like [`LanczosFactorization`](@ref). The block tridiagonal matrix `TDB` is preallocated in `BlockLanczosFactorization`
and is of type `Hermitian{S<:Number}` with `size(TDB) == (k,k)`. The residuals `r` is of type `Vector{T}`.
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
    total_size::Int
    const V::OrthonormalBasis{T}      # BlockLanczos Basis
    const TDB::AbstractMatrix{S}      # TDB matrix, S is the matrix type
    const r::BlockVec{T,S}            # residual block
    r_size::Int # size of the residual block
    norm_r::SR  # norm of the residual block
end
Base.length(fact::BlockLanczosFactorization) = fact.total_size
normres(fact::BlockLanczosFactorization) = fact.norm_r
basis(fact::BlockLanczosFactorization) = fact.V
residual(fact::BlockLanczosFactorization) = fact.r[1:(fact.r_size)]

"""
    struct BlockLanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    BlockLanczosIterator(f, x₀, maxdim, qr_tol, [orth::Orthogonalizer = KrylovDefaults.orth])

Iterator that takes a linear map `f::F` (supposed to be real symmetric or complex hermitian)
and an initial block `x₀::BlockVec{T,S}` and generates an expanding `BlockLanczosFactorization` thereof. In
particular, `BlockLanczosIterator` uses the
[BlockLanczos iteration](https://en.wikipedia.org/wiki/Block_Lanczos_algorithm) scheme to build a
successively expanding BlockLanczos factorization. While `f` cannot be tested to be symmetric or
hermitian directly when the linear map is encoded as a general callable object or function, with `block_inner(X, f.(X))`,
it is tested whether `norm(M-M')` is sufficiently small to be neglected.

The argument `f` can be a matrix, or a function accepting a single argument `x`, so that
`f(x)` implements the action of the linear map on the block `x`.

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
    TDB = zeros(S, maxdim, maxdim)

    # Orthogonalization of the initial block
    X₁ = deepcopy(X₀)
    abstract_qr!(X₁, iter.qr_tol)
    V = OrthonormalBasis(X₁.vec)

    AX₁ = apply(A, X₁)
    M₁ = block_inner(X₁, AX₁)
    TDB[1:bs, 1:bs] .= M₁
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
                                     TDB,
                                     AX₁,
                                     bs,
                                     norm_r)
end

function expand!(iter::BlockLanczosIterator{F,T,S},
                 state::BlockLanczosFactorization{T,S,SR};
                 verbosity::Int=KrylovDefaults.verbosity[]) where {F,T,S,SR}
    k = state.total_size
    rₖ = state.r[1:(state.r_size)]
    bs_now = length(rₖ)
    V = state.V

    # Calculate the new basis and Bₖ
    Bₖ, good_idx = abstract_qr!(rₖ, iter.qr_tol)
    bs_next = length(good_idx)
    push!(V, rₖ[good_idx])
    state.TDB[(k + 1):(k + bs_next), (k - bs_now + 1):k] .= Bₖ
    state.TDB[(k - bs_now + 1):k, (k + 1):(k + bs_next)] .= Bₖ'

    # Calculate the new residual and orthogonalize the new basis
    rₖnext, Mnext = blocklanczosrecurrence(iter.operator, V, Bₖ, iter.orth)
    verbosity >= WARN_LEVEL && warn_nonhermitian(Mnext)

    state.TDB[(k + 1):(k + bs_next), (k + 1):(k + bs_next)] .= Mnext
    state.r.vec[1:bs_next] .= rₖnext.vec
    state.norm_r = norm(rₖnext)
    state.total_size += bs_next
    state.r_size = bs_next

    if verbosity > EACHITERATION_LEVEL
        @info "BlockLanczos expansion to dimension $(state.total_size): subspace normres = $(normres2string(state.norm_r))"
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
    rₖnext = compute_residual!(AX, X, M, Xlast, Bₖ)
    ortho_basis!(rₖnext, V)
    return rₖnext, M
end

"""
    compute_residual!(AX::BlockVec{T,S}, X::BlockVec{T,S},
                           M::AbstractMatrix,
                           X_prev::BlockVec{T,S}, B_prev::AbstractMatrix) where {T,S}

This function computes the residual vector `AX` by subtracting the operator applied to `X` from `AX`,
and then subtracting the projection of `AX` onto the previously orthonormalized basis vectors in `X_prev`.
The result is stored in place in `AX`.

```
    AX <- AX - X * M - X_prev * B_prev
```

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
    ortho_basis!(basis::BlockVec{T,S}, basis_sofar::OrthonormalBasis{T}) where {T,S}

This function orthogonalizes the vectors in `basis` with respect to the previously orthonormalized set `basis_sofar` by using the modified Gram-Schmidt process.
Specifically, it modifies each vector `basis[i]` by projecting out its components along the directions spanned by `basis_sofar`, i.e.,

```
    basis[i] = basis[i] - sum(j=1:length(basis_sofar)) <basis[i], basis_sofar[j]> basis_sofar[j]
```

Here,`⟨·,·⟩` denotes the inner product. The function assumes that `basis_sofar` is already orthonormal.
"""
function ortho_basis!(basis::BlockVec{T,S}, basis_sofar::OrthonormalBasis{T}) where {T,S}
    for i in 1:length(basis)
        for q in basis_sofar
            basis[i], _ = orthogonalize!!(basis[i], q, ModifiedGramSchmidt())
        end
    end
    return basis
end

function warn_nonhermitian(M::AbstractMatrix)
    if norm(M - M') > eps(real(eltype(M)))^(2 / 5)
        @warn "Enforce Hermiticity on the triangular diagonal blocks matrix, even though the operator may not be Hermitian."
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
            block[j] = scale(block[j], 1 / β)
        else
            block[j] = scale!!(block[j], 0)
            rank_shrink = true
            idx[j] = 0
        end
    end
    good_idx = findall(idx .> 0)
    return R[good_idx, :], good_idx
end
