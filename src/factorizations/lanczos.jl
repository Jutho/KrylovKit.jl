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

function initialize(iter::LanczosIterator; verbosity::Int=0)
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
    warn_nonhermitian(α, zero(β), β)
    # possibly reorthogonalize
    if iter.orth isa Union{ClassicalGramSchmidt2,ModifiedGramSchmidt2}
        dα = inner(v, r)
        α += dα
        r = add!!(r, v, -dα) # should we use real(dα) here?
        β = norm(r)
        warn_nonhermitian(α, zero(β), β)
    elseif iter.orth isa Union{ClassicalGramSchmidtIR,ModifiedGramSchmidtIR}
        while eps(one(β)) < β < iter.orth.η * βold
            βold = β
            dα = inner(v, r)
            α += dα
            r = add!!(r, v, -dα) # should we use real(dα) here?
            β = norm(r)
            warn_nonhermitian(α, zero(β), β)
        end
    end
    V = OrthonormalBasis([v])
    αs = [real(α)]
    βs = [β]
    if verbosity > 0
        @info "Lanczos iteration step 1: normres = $β"
    end
    return LanczosFactorization(1, V, αs, βs, r)
end
function initialize!(iter::LanczosIterator, state::LanczosFactorization; verbosity::Int=0)
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
    warn_nonhermitian(α, zero(β), β)

    state.k = 1
    push!(αs, real(α))
    push!(βs, β)
    state.r = r
    if verbosity > 0
        @info "Lanczos iteration step 1: normres = $β"
    end
    return state
end
function expand!(iter::LanczosIterator, state::LanczosFactorization; verbosity::Int=0)
    βold = normres(state)
    V = state.V
    r = state.r
    V = push!(V, scale!!(r, 1 / βold))
    r, α, β = lanczosrecurrence(iter.operator, V, βold, iter.orth)
    warn_nonhermitian(α, βold, β)

    αs = push!(state.αs, real(α))
    βs = push!(state.βs, β)

    !iter.keepvecs && popfirst!(state.V) # remove oldest V if not keepvecs

    state.k += 1
    state.r = r
    if verbosity > 0
        @info "Lanczos iteration step $(state.k): normres = $β"
    end
    return state
end
function shrink!(state::LanczosFactorization, k)
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
    state.r = scale!!(r, normres(state))
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
