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
length 1, and `expand!(::KrylovIterator, ::KrylovFactorization)`(@ref) expands the
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


# block lanczos

#= 
The basic theory of the Block Lanczos algorithm can be referred to : 
Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed., pp. 566–569). Johns Hopkins University Press.

Now what I implement is block lanczos with mutable block size. But I'm still confused is it neccesary. That is to say, Can we asseert 
the iteration would end with size shrink? 
Mathematically: for a set of initial abstract vectors X₀ = {x₁,..,xₚ}, where A is a hermitian operator, if 
Sₖ = {x ∈ AʲX₀:j=0,..,k-1}
is linear dependent, can we assert that Rₖ ∈ span(A^{k-2}X₀,A^{k-1}X₀) or at least in span(Sₖ)?
For vectors in F^d I believe it's right. But in a abstract inner product space, it's obviouly much more complicated.

What ever, mutable block size is at least undoubtedly useful for non-hermitian operator so I implement it.
https://www.netlib.org/utk/people/JackDongarra/etemplates/node252.html#ABLEsection
=#

mutable struct BlockLanczosFactorization{T,S<:Number,SR<:Real} <: BlockKrylovFactorization{T,S,SR}
    all_size::Int
    const V::OrthonormalBasis{T}            # Block Lanczos Basis
    const TDB::AbstractMatrix{S}            # TDB matrix, S is the matrix type
    const r::OrthonormalBasis{T}            # residual block
    r_size::Int
    norm_r::SR
end

#= 
Now our orthogonalizer is only ModifiedGramSchmidt2.
Dimension of Krylov subspace in BlockLanczosIterator is usually much bigger than lanczos.
So ClassicalGramSchmidt and ModifiedGramSchmidt1 is numerically unstable.
I don't add IR orthogonalizer because I find it sometimes unstable and I am studying it.
Householder reorthogonalization is theoretically stable and saves memory, but the algorithm I implemented is not stable.
In the future, I will add IR and Householder orthogonalizer.
=#

#= The only orthogonalization method we use in block lanczos is ModifiedGramSchmidt2. So we don't need to
provide "keepvecs" because we have to reverse all krylove vectors.
=#
struct BlockLanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::Vector{T}
    maxdim::Int
    num_field::Type
    orth::O
    qr_tol::Real
    function BlockLanczosIterator{F,T,O}(operator::F,
                                         x₀::Vector{T},
                                         maxdim::Int,
                                         num_field::Type,
                                         orth::O,
                                         qr_tol::Real) where {F,T,O<:Orthogonalizer}
        return new{F,T,O}(operator, x₀, maxdim, num_field, orth, qr_tol)
    end
end
function BlockLanczosIterator(operator::F,
                              x₀::AbstractVector{T},
                              maxdim::Int,
                              qr_tol::Real,
                              orth::O=ModifiedGramSchmidt2()) where {F,T,O<:Orthogonalizer}
    S = typeof(inner(x₀[1], x₀[1]))
    qr_tol < 0 && (qr_tol = 1e4 * eps(real(S)))
    length(x₀) < 2 && @error "initial vector should not have norm zero"
    norm(x₀) < qr_tol && @error "initial vector should not have norm zero"
    orth != ModifiedGramSchmidt2() && @error "BlockLanczosIterator only supports ModifiedGramSchmidt2 orthogonalizer"
    return BlockLanczosIterator{F,T,O}(operator, x₀, maxdim, S, orth, qr_tol)
end 

function initialize(iter::BlockLanczosIterator; verbosity::Int=KrylovDefaults.verbosity[])
    x₀_vec = iter.x₀
    maxdim = iter.maxdim
    bs_now = length(x₀_vec) # block size now
    A = iter.operator
    S = iter.num_field

    V_basis = [similar(x₀_vec[1]) for i in 1:bs_now * (maxdim + 1)]
    r = [similar(x₀_vec[i]) for i in 1:bs_now]
    TDB = zeros(S, bs_now * (maxdim + 1), bs_now * (maxdim + 1))

    X₁_view = view(V_basis, 1:bs_now)
    copy!.(X₁_view, x₀_vec)
    # Orthogonalization of the initial block
    abstract_qr!(BlockVec(X₁_view, S), iter.qr_tol)
    Ax₁ = [apply(A, x) for x in X₁_view]
    M₁_view = view(TDB, 1:bs_now, 1:bs_now)
    block_inner!(M₁_view, BlockVec(X₁_view, S), BlockVec(Ax₁, S))
    verbosity >= WARN_LEVEL && warn_nonhermitian(M₁_view, iter.qr_tol)
    M₁_view = (M₁_view + M₁_view') / 2
 
    residual = block_mul!(BlockVec(Ax₁, S), BlockVec(X₁_view, S), - M₁_view, 1.0, 1.0).vec

    # QR decomposition of residual to get the next basis. Get X2 and B1
    B₁, good_idx = abstract_qr!(BlockVec(residual, S), iter.qr_tol)
    bs_next = length(good_idx)
    X₂_view = view(V_basis, bs_now+1:bs_now+bs_next)
    copy!.(X₂_view, residual[good_idx])
    B₁_view = view(TDB, bs_now+1:bs_now+bs_next, 1:bs_now)
    copyto!(B₁_view, B₁)
    copyto!(view(TDB, 1:bs_now, bs_now+1:bs_now+bs_next), B₁_view')

    # Calculate the next block
    Ax₂ = [apply(A, x) for x in X₂_view]
    M₂_view = view(TDB, bs_now+1:bs_now+bs_next, bs_now+1:bs_now+bs_next)
    block_inner!(M₂_view, BlockVec(X₂_view, S), BlockVec(Ax₂, S))
    M₂_view = (M₂_view + M₂_view') / 2

    # Calculate the new residual. Get r₂
    compute_residual!(BlockVec(r, S), BlockVec(Ax₂, S), BlockVec(X₂_view, S), M₂_view, BlockVec(X₁_view, S), B₁_view)
    ortho_basis!(BlockVec(r, S), BlockVec(view(V_basis, 1:bs_now+bs_next), S))

    norm_r = norm(r)

    if verbosity > EACHITERATION_LEVEL
        @info "Block Lanczos initiation at dimension 2: subspace normres = $(normres2string(norm_r))"
    end

    return BlockLanczosFactorization(bs_now+bs_next,
                                    OrthonormalBasis(V_basis),
                                    TDB,
                                    OrthonormalBasis(r),
                                    bs_next,
                                    norm_r)
end

function expand!(iter::BlockLanczosIterator, state::BlockLanczosFactorization;
                 verbosity::Int=KrylovDefaults.verbosity[])
    all_size = state.all_size
    rₖ = view(state.r.basis, 1:state.r_size)
    S = iter.num_field
    bs_now = length(rₖ)

    # Get the current residual as the initial value of the new basis. Get Xnext
    Bₖ, good_idx = abstract_qr!(BlockVec(rₖ, S), iter.qr_tol)
    bs_next = length(good_idx)
    Xnext_view = view(state.V.basis, all_size+1:all_size+bs_next)
    copy!.(Xnext_view, rₖ[good_idx])

    # Calculate the connection matrix
    Bₖ_view = view(state.TDB, all_size+1:all_size+bs_next, all_size-bs_now+1:all_size)
    copyto!(Bₖ_view, Bₖ)
    copyto!(view(state.TDB, all_size-bs_now+1:all_size, all_size+1:all_size+bs_next), Bₖ_view')

    # Apply the operator and calculate the M. Get Mnext
    Axₖnext = [apply(iter.operator, x) for x in Xnext_view]
    Mnext_view = view(state.TDB, all_size+1:all_size+bs_next, all_size+1:all_size+bs_next)
    block_inner!(Mnext_view, BlockVec(Xnext_view, S), BlockVec(Axₖnext, S))
    verbosity >= WARN_LEVEL && warn_nonhermitian(Mnext_view, iter.qr_tol)
    Mnext_view = (Mnext_view + Mnext_view') / 2

    # Calculate the new residual. Get Rnext
    Xnow_view = view(state.V.basis, all_size-bs_now+1:all_size)
    rₖ[1:bs_next] = rₖ[good_idx]
    rₖnext_view = view(state.r.basis, 1:bs_next)
    compute_residual!(BlockVec(rₖnext_view, S), BlockVec(Axₖnext, S), BlockVec(Xnext_view, S), Mnext_view, BlockVec(Xnow_view, S), Bₖ_view)
    ortho_basis!(BlockVec(rₖnext_view, S), BlockVec(view(state.V.basis, 1:all_size+bs_next), S))
    state.norm_r = norm(rₖnext_view)
    state.all_size += bs_next
    state.r_size = bs_next

    if verbosity > EACHITERATION_LEVEL
        @info "Block Lanczos expansion to dimension $(state.all_size): subspace normres = $(normres2string(state.norm_r))"
    end
end

function compute_residual!(Block_r::BlockVec{T,S}, Block_A_X::BlockVec{T,S}, Block_X::BlockVec{T,S}, M::AbstractMatrix, Block_X_prev::BlockVec{T,S}, B_prev::AbstractMatrix) where {T,S}
    r = Block_r.vec
    A_X = Block_A_X.vec
    X = Block_X.vec
    X_prev = Block_X_prev.vec
    @inbounds for j in 1:length(X)
        r_j = r[j] 
        copy!(r_j, A_X[j])
        for i in 1:length(X)
            axpy!(- M[i,j], X[i], r_j)
        end
        for i in 1:length(X_prev)
            axpy!(- B_prev[i,j], X_prev[i], r_j)
        end
    end
    return Block_r
end

function ortho_basis!(basis_new::BlockVec{T,S}, basis_sofar::BlockVec{T,S}) where {T,S}
    tmp = Matrix{S}(undef, length(basis_sofar), length(basis_new))
    block_inner!(tmp, basis_sofar, basis_new)
    block_mul!(basis_new, basis_sofar, - tmp, 1.0, 1.0)
    return basis_new
end

function warn_nonhermitian(M::AbstractMatrix, tol::Real)
    if norm(M - M') > tol
        @warn "Enforce Hermiticity on the triangular diagonal blocks matrix, even though the operator may not be Hermitian."
    end
end

function abstract_qr!(Block::BlockVec{T,S}, tol::Real) where {T,S}
    block = Block.vec
    n = length(block)
    rank_shrink = false
    idx = ones(Int64,n)
    R = zeros(S, n, n)
    @inbounds for j in 1:n
        αⱼ = block[j]
        for i in 1:j-1
            R[i, j] = inner(block[i], αⱼ)
            αⱼ -= R[i, j] * block[i]
        end
        β = norm(αⱼ)
        if !(β ≤ tol)
            R[j, j] = β
            block[j] = αⱼ / β
        else
            block[j] *= S(0)
            rank_shrink = true
            idx[j] = 0
        end
    end
    good_idx = findall(idx .> 0)
    return R[good_idx,:], good_idx
end

function shrink!(state::BlockLanczosFactorization, k; verbosity::Int=KrylovDefaults.verbosity[])
    @show "shrink!"
    V = state.V
    all_size = state.all_size
    V[1:k] = V[all_size-k+1:all_size]
    newTDB = zero(state.TDB)
    newTDB[1:k,1:k] = state.TDB[all_size-k+1:all_size,all_size-k+1:all_size]
    state.all_size = k
    state.TDB .= newTDB
    if verbosity > EACHITERATION_LEVEL
        @info "Lanczos reduction to dimension $k"
    end
    return state
end