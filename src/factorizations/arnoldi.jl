# arnoldi.jl
"""
    mutable struct ArnoldiFactorization{T,S} <: KrylovFactorization{T,S}

Structure to store an Arnoldi factorization of a linear map `A` of the form

```julia
A * V = V * B + r * b'
```

For a given Arnoldi factorization `fact` of length `k = length(fact)`, the basis `V` is
obtained via [`basis(fact)`](@ref basis) and is an instance of [`OrthonormalBasis{T}`](@ref
Basis), with also `length(V) == k` and where `T` denotes the type of vector like objects
used in the problem. The Rayleigh quotient `B` is obtained as
[`rayleighquotient(fact)`](@ref) and is of type [`B::PackedHessenberg{S<:Number}`](@ref
PackedHessenberg) with `size(B) == (k,k)`. The residual `r` is obtained as
[`residual(fact)`](@ref) and is of type `T`. One can also query [`normres(fact)`](@ref) to
obtain `norm(r)`, the norm of the residual. The vector `b` has no dedicated name but can be
obtained via [`rayleighextension(fact)`](@ref). It takes the default value ``e_k``, i.e. the
unit vector of all zeros and a one in the last entry, which is represented using
[`SimpleBasisVector`](@ref).

An Arnoldi factorization `fact` can be destructured as `V, B, r, nr, b = fact` with
`nr = norm(r)`.

`ArnoldiFactorization` is mutable because it can [`expand!`](@ref) or [`shrink!`](@ref).
See also [`ArnoldiIterator`](@ref) for an iterator that constructs a progressively expanding
Arnoldi factorizations of a given linear map and a starting vector. See
[`LanczosFactorization`](@ref) and [`LanczosIterator`](@ref) for a Krylov factorization that
is optimized for real symmetric or complex hermitian linear maps.
"""
mutable struct ArnoldiFactorization{T,S} <: KrylovFactorization{T,S}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    H::Vector{S} # stores the Hessenberg matrix in packed form
    r::T # residual
end

Base.length(F::ArnoldiFactorization) = F.k
Base.sizehint!(F::ArnoldiFactorization, n) = begin
    sizehint!(F.V, n)
    sizehint!(F.H, (n * n + 3 * n) >> 1)
    return F
end
Base.eltype(F::ArnoldiFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:ArnoldiFactorization{<:Any,S}}) where {S} = S

basis(F::ArnoldiFactorization) = F.V
rayleighquotient(F::ArnoldiFactorization) = PackedHessenberg(F.H, F.k)
residual(F::ArnoldiFactorization) = F.r
@inbounds normres(F::ArnoldiFactorization) = abs(F.H[end])
rayleighextension(F::ArnoldiFactorization) = SimpleBasisVector(F.k, F.k)

# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
"""
    struct ArnoldiIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    ArnoldiIterator(f, v₀, [orth::Orthogonalizer = KrylovDefaults.orth])

Iterator that takes a general linear map `f::F` and an initial vector `v₀::T` and generates
an expanding `ArnoldiFactorization` thereof. In particular, `ArnoldiIterator` iterates over
progressively expanding Arnoldi factorizations using the
[Arnoldi iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration).

The argument `f` can be a matrix, or a function accepting a single argument `v`, so that
`f(v)` implements the action of the linear map on the vector `v`.

The optional argument `orth` specifies which [`Orthogonalizer`](@ref) to be used. The
default value in [`KrylovDefaults`](@ref) is to use [`ModifiedGramSchmidtIR`](@ref), which
possibly uses reorthogonalization steps.

When iterating over an instance of `ArnoldiIterator`, the values being generated are
instances of [`ArnoldiFactorization`](@ref), which can be immediately destructured into a
[`basis`](@ref), [`rayleighquotient`](@ref), [`residual`](@ref), [`normres`](@ref) and
[`rayleighextension`](@ref), for example as

```julia
for (V, B, r, nr, b) in ArnoldiIterator(f, v₀)
    # do something
    nr < tol && break # a typical stopping criterion
end
```

Since the iterator does not know the dimension of the underlying vector space of
objects of type `T`, it keeps expanding the Krylov subspace until the residual norm `nr`
falls below machine precision `eps(typeof(nr))`.

The internal state of `ArnoldiIterator` is the same as the return value, i.e. the
corresponding `ArnoldiFactorization`. However, as Julia's Base iteration interface (using
`Base.iterate`) requires that the state is not mutated, a `deepcopy` is produced upon every
next iteration step.

Instead, you can also mutate the `ArnoldiFactorization` in place, using the following
interface, e.g. for the same example above

```julia
iterator = ArnoldiIterator(f, v₀)
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
struct ArnoldiIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::T
    orth::O
end
ArnoldiIterator(A, x₀) = ArnoldiIterator(A, x₀, KrylovDefaults.orth)

Base.IteratorSize(::Type{<:ArnoldiIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:ArnoldiIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::ArnoldiIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::ArnoldiIterator, state)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::ArnoldiIterator; verbosity::Int=0)
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
    r = add!!(r, v, -α)
    β = norm(r)
    # possibly reorthogonalize
    if iter.orth isa Union{ClassicalGramSchmidt2,ModifiedGramSchmidt2}
        dα = inner(v, r)
        α += dα
        r = add!!(r, v, -dα)
        β = norm(r)
    elseif iter.orth isa Union{ClassicalGramSchmidtIR,ModifiedGramSchmidtIR}
        while eps(one(β)) < β < iter.orth.η * βold
            βold = β
            dα = inner(v, r)
            α += dα
            r = add!!(r, v, -dα)
            β = norm(r)
        end
    end
    V = OrthonormalBasis([v])
    H = T[α, β]
    if verbosity > 0
        @info "Arnoldi iteration step 1: normres = $β"
    end
    return state = ArnoldiFactorization(1, V, H, r)
end
function initialize!(iter::ArnoldiIterator, state::ArnoldiFactorization; verbosity::Int=0)
    x₀ = iter.x₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    H = empty!(state.H)

    V[1] = scale!!(V[1], x₀, 1 / norm(x₀))
    w = apply(iter.operator, V[1])
    r, α = orthogonalize!!(w, V[1], iter.orth)
    β = norm(r)
    state.k = 1
    push!(H, α, β)
    state.r = r
    if verbosity > 0
        @info "Arnoldi iteration step 1: normres = $β"
    end
    return state
end
function expand!(iter::ArnoldiIterator, state::ArnoldiFactorization; verbosity::Int=0)
    state.k += 1
    k = state.k
    V = state.V
    H = state.H
    r = state.r
    β = normres(state)
    push!(V, scale(r, 1 / β))
    m = length(H)
    resize!(H, m + k + 1)
    r, β = arnoldirecurrence!!(iter.operator, V, view(H, (m + 1):(m + k)), iter.orth)
    H[m + k + 1] = β
    state.r = r
    if verbosity > 0
        @info "Arnoldi iteration step $k: normres = $β"
    end
    return state
end
function shrink!(state::ArnoldiFactorization, k)
    length(state) <= k && return state
    V = state.V
    H = state.H
    while length(V) > k + 1
        pop!(V)
    end
    r = pop!(V)
    resize!(H, (k * k + 3 * k) >> 1)
    state.k = k
    state.r = scale!!(r, normres(state))
    return state
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!!(operator,
                             V::OrthonormalBasis,
                             h::AbstractVector,
                             orth::Orthogonalizer)
    w = apply(operator, last(V))
    r, h = orthogonalize!!(w, V, h, orth)
    return r, norm(r)
end
