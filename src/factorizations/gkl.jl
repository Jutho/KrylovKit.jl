# gkl.jl
"""
    mutable struct GKLFactorization{TU,TV,S<:Real}

Structure to store a Golub-Kahan-Lanczos (GKL) bidiagonal factorization of a linear map `A`
of the form

```julia
A * V = U * B + r * b'
A' * U = V * B'
```

For a given GKL factorization `fact` of length `k = length(fact)`, the two bases `U` and `V`
are obtained via [`basis(fact, :U)`](@ref basis) and `basis(fact, :V)`. Here, `U` and `V`
are instances of [`OrthonormalBasis{T}`](@ref Basis), with also
`length(U) == length(V) == k` and where `T` denotes the type of vector like objects used in
the problem. The Rayleigh quotient `B` is obtained as [`rayleighquotient(fact)`](@ref) and
is of type `Bidiagonal{S<:Number}` with `size(B) == (k,k)`. The residual `r` is
obtained as [`residual(fact)`](@ref) and is of type `T`. One can also query
[`normres(fact)`](@ref) to obtain `norm(r)`, the norm of the residual. The vector `b` has no
dedicated name but can be obtained via [`rayleighextension(fact)`](@ref). It takes the
default value ``e_k``, i.e. the unit vector of all zeros and a one in the last entry, which
is represented using [`SimpleBasisVector`](@ref).

A GKL factorization `fact` can be destructured as `U, V, B, r, nr, b = fact` with
`nr = norm(r)`.

`GKLFactorization` is mutable because it can [`expand!`](@ref) or [`shrink!`](@ref).
See also [`GKLIterator`](@ref) for an iterator that constructs a progressively expanding
GKL factorizations of a given linear map and a starting vector `u₀`.
"""
mutable struct GKLFactorization{TU,TV,S<:Real}
    k::Int # current Krylov dimension
    U::OrthonormalBasis{TU} # basis of length k
    V::OrthonormalBasis{TV} # basis of length k
    αs::Vector{S}
    βs::Vector{S}
    r::TU
end

Base.length(F::GKLFactorization) = F.k
Base.sizehint!(F::GKLFactorization, n) = begin
    sizehint!(F.U, n)
    sizehint!(F.V, n)
    sizehint!(F.αs, n)
    sizehint!(F.βs, n)
    return F
end
Base.eltype(F::GKLFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:GKLFactorization{<:Any,<:Any,S}}) where {S} = S

# iteration for destructuring into components
Base.iterate(F::GKLFactorization) = (basis(F, :U), Val(:V))
Base.iterate(F::GKLFactorization, ::Val{:V}) = (basis(F, :V), Val(:rayleighquotient))
function Base.iterate(F::GKLFactorization, ::Val{:rayleighquotient})
    return (rayleighquotient(F), Val(:residual))
end
Base.iterate(F::GKLFactorization, ::Val{:residual}) = (residual(F), Val(:normres))
Base.iterate(F::GKLFactorization, ::Val{:normres}) = (normres(F), Val(:rayleighextension))
function Base.iterate(F::GKLFactorization, ::Val{:rayleighextension})
    return (rayleighextension(F), Val(:done))
end
Base.iterate(F::GKLFactorization, ::Val{:done}) = nothing

"""
        basis(fact::GKLFactorization, which::Symbol)

Return the list of basis vectors of a [`GKLFactorization`](@ref), where `which` should take
the value `:U` or `:V` and indicates which set of basis vectors (in the domain or in the
codomain of the corresponding linear map) should be returned. The return type is an
`OrthonormalBasis{T}`, where `T` represents the type of the vectors used by the problem.
"""
function basis(F::GKLFactorization, which::Symbol)
    length(F.U) == F.k || error("Not keeping vectors during GKL bidiagonalization")
    which == :U || which == :V || error("invalid flag for specifying basis")
    return which == :U ? F.U : F.V
end
function rayleighquotient(F::GKLFactorization)
    return Bidiagonal(view(F.αs, 1:(F.k)), view(F.βs, 1:(F.k - 1)), :L)
end
residual(F::GKLFactorization) = F.r
@inbounds normres(F::GKLFactorization) = F.βs[F.k]
rayleighextension(F::GKLFactorization) = SimpleBasisVector(F.k, F.k)

# GKL iteration for constructing the orthonormal basis of a Krylov subspace.
"""
    struct GKLIterator{F,TU,O<:Orthogonalizer}
    GKLIterator(f, u₀, [orth::Orthogonalizer = KrylovDefaults.orth, keepvecs::Bool = true])

Iterator that takes a general linear map `f::F` and an initial vector `u₀::TU` and generates
an expanding `GKLFactorization` thereof. In particular, `GKLIterator` implements the
[Golub-Kahan-Lanczos bidiagonalization procedure](http://www.netlib.org/utk/people/JackDongarra/etemplates/node198.html).
Note, however, that this implementation starts from a vector `u₀` in the codomain of the
linear map `f`, which will end up (after normalisation) as the first column of `U`.

The argument `f` can be a matrix, a tuple of two functions where the first represents the
normal action and the second the adjoint action, or a function accepting two arguments,
where the first argument is the vector to which the linear map needs to be applied, and the
second argument is either `Val(false)` for the normal action and `Val(true)` for the adjoint
action. Note that the flag is thus a `Val` type to allow for type stability in cases where
the vectors in the domain and the codomain of the linear map have a different type.

The optional argument `orth` specifies which [`Orthogonalizer`](@ref) to be used. The
default value in [`KrylovDefaults`](@ref) is to use [`ModifiedGramSchmidtIR`](@ref), which
possibly uses reorthogonalization steps.

When iterating over an instance of `GKLIterator`, the values being generated are
instances `fact` of [`GKLFactorization`](@ref), which can be immediately destructured into a
[`basis(fact, :U)`](@ref), [`basis(fact, :V)`](@ref), [`rayleighquotient`](@ref),
[`residual`](@ref), [`normres`](@ref) and [`rayleighextension`](@ref), for example as

```julia
for (U, V, B, r, nr, b) in GKLIterator(f, u₀)
    # do something
    nr < tol && break # a typical stopping criterion
end
```

Since the iterator does not know the dimension of the underlying vector space of
objects of type `T`, it keeps expanding the Krylov subspace until the residual norm `nr`
falls below machine precision `eps(typeof(nr))`.

The internal state of `GKLIterator` is the same as the return value, i.e. the corresponding
`GKLFactorization`. However, as Julia's Base iteration interface (using `Base.iterate`)
requires that the state is not mutated, a `deepcopy` is produced upon every next iteration
step.

Instead, you can also mutate the `GKLFactorization` in place, using the following
interface, e.g. for the same example above

```julia
iterator = GKLIterator(f, u₀)
factorization = initialize(iterator)
while normres(factorization) > tol
    expand!(iterator, factorization)
    U, V, B, r, nr, b = factorization
    # do something
end
```

Here, [`initialize(::GKLIterator)`](@ref) produces the first GKL factorization of length 1,
and `expand!(::GKLIterator, ::GKLFactorization)`(@ref) expands the factorization in place.
See also [`initialize!(::GKLIterator, ::GKLFactorization)`](@ref) to initialize in an
already existing factorization (most information will be discarded) and
[`shrink!(::GKLIterator, k)`](@ref) to shrink an existing factorization down to length `k`.
"""
struct GKLIterator{F,TU,O<:Orthogonalizer}
    operator::F
    u₀::TU
    orth::O
    keepvecs::Bool
    function GKLIterator{F,TU,O}(operator::F,
                                 u₀::TU,
                                 orth::O,
                                 keepvecs::Bool) where {F,TU,O<:Orthogonalizer}
        if !keepvecs && isa(orth, Reorthogonalizer)
            error("Cannot use reorthogonalization without keeping all Krylov vectors")
        end
        return new{F,TU,O}(operator, u₀, orth, keepvecs)
    end
end
function GKLIterator(operator::F,
                     u₀::TU,
                     orth::O=KrylovDefaults.orth,
                     keepvecs::Bool=true) where {F,TU,O<:Orthogonalizer}
    return GKLIterator{F,TU,O}(operator, u₀, orth, keepvecs)
end

Base.IteratorSize(::Type{<:GKLIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:GKLIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::GKLIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::GKLIterator, state::GKLFactorization)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::GKLIterator; verbosity::Int=0)
    # initialize without using eltype
    u₀ = iter.u₀
    β₀ = norm(u₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    v₀ = apply_adjoint(iter.operator, u₀)
    α = norm(v₀) / β₀
    Av₀ = apply_normal(iter.operator, v₀) # apply operator
    α² = inner(u₀, Av₀) / β₀^2
    α² ≈ α * α || throw(ArgumentError("operator and its adjoint are not compatible"))
    T = typeof(α²) # scalar type of the Rayleigh quotient

    # these lines determines the vector types that we will henceforth use
    u = scale(u₀, one(T) / β₀)
    v = scale(v₀, one(T) / (α * β₀))
    if typeof(Av₀) == typeof(u)
        r = scale!!(Av₀, 1 / (α * β₀))
    else
        r = scale!!(zerovector(u), Av₀, 1 / (α * β₀))
    end
    r = add!!(r, u, -α)
    β = norm(r)

    U = OrthonormalBasis([u])
    V = OrthonormalBasis([v])
    S = real(T)
    αs = S[α]
    βs = S[β]
    if verbosity > 0
        @info "GKL iteration step 1: normres = $β"
    end

    return GKLFactorization(1, U, V, αs, βs, r)
end
function initialize!(iter::GKLIterator, state::GKLFactorization; verbosity::Int=0)
    U = state.U
    while length(U) > 1
        pop!(U)
    end
    V = empty!(state.V)
    αs = empty!(state.αs)
    βs = empty!(state.βs)

    u = scale!!(U[1], iter.u₀, 1 / norm(iter.u₀))
    v = apply_adjoint(iter.operator, u)
    α = norm(v)
    v = scale!!(v, inv(α))
    r = apply_normal(iter.operator, v)
    r = add!!(r, u, -α)
    β = norm(r)

    state.k = 1
    push!(V, v)
    push!(αs, α)
    push!(βs, β)
    state.r = r
    if verbosity > 0
        @info "GKL iteration step 1: normres = $β"
    end

    return state
end
function expand!(iter::GKLIterator, state::GKLFactorization; verbosity::Int=0)
    βold = normres(state)
    U = state.U
    V = state.V
    r = state.r
    U = push!(U, scale!!(r, 1 / βold))
    v, r, α, β = gklrecurrence(iter.operator, U, V, βold, iter.orth)

    push!(V, v)
    push!(state.αs, α)
    push!(state.βs, β)

    #!iter.keepvecs && popfirst!(state.V) # remove oldest V if not keepvecs

    state.k += 1
    state.r = r
    if verbosity > 0
        @info "GKL iteration step $(state.k): normres = $β"
    end

    return state
end
function shrink!(state::GKLFactorization, k)
    length(state) == length(state.V) ||
        error("we cannot shrink GKLFactorization without keeping vectors")
    length(state) <= k && return state
    U = state.U
    V = state.V
    while length(V) > k + 1
        pop!(U)
        pop!(V)
    end
    pop!(V)
    r = pop!(U)
    resize!(state.αs, k)
    resize!(state.βs, k)
    state.k = k
    state.r = scale!!(r, normres(state))
    return state
end

# Golub-Kahan-Lanczos recurrence relation
function gklrecurrence(operator,
                       U::OrthonormalBasis,
                       V::OrthonormalBasis,
                       β,
                       orth::Union{ClassicalGramSchmidt,ModifiedGramSchmidt})
    u = U[end]
    v = apply_adjoint(operator, u)
    v = add!!(v, V[end], -β)
    α = norm(v)
    v = scale!!(v, inv(α))

    r = apply_normal(operator, v)
    r = add!!(r, u, -α)
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(operator,
                       U::OrthonormalBasis,
                       V::OrthonormalBasis,
                       β,
                       orth::ClassicalGramSchmidt2)
    u = U[end]
    v = apply_adjoint(operator, u)
    v = add!!(v, V[end], -β) # not necessary if we definitely reorthogonalize next step and previous step
    # v, = orthogonalize!(v, V, ClassicalGramSchmidt())
    α = norm(v)
    v = scale!!(v, inv(α))

    r = apply_normal(operator, v)
    r = add!!(r, u, -α)
    r, = orthogonalize!!(r, U, ClassicalGramSchmidt())
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(operator,
                       U::OrthonormalBasis,
                       V::OrthonormalBasis,
                       β,
                       orth::ModifiedGramSchmidt2)
    u = U[end]
    v = apply_adjoint(operator, u)
    v = add!!(v, V[end], -β)
    # for q in V # not necessary if we definitely reorthogonalize next step and previous step
    #     v, = orthogonalize!(v, q, ModifiedGramSchmidt())
    # end
    α = norm(v)
    v = scale!!(v, inv(α))

    r = apply_normal(operator, v)
    r = add!!(r, u, -α)
    for q in U
        r, = orthogonalize!!(r, q, ModifiedGramSchmidt())
    end
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(operator,
                       U::OrthonormalBasis,
                       V::OrthonormalBasis,
                       β,
                       orth::ClassicalGramSchmidtIR)
    u = U[end]
    v = apply_adjoint(operator, u)
    v = add!!(v, V[end], -β)
    α = norm(v)
    nold = sqrt(abs2(α) + abs2(β))
    while α < orth.η * nold
        nold = α
        v, = orthogonalize!!(v, V, ClassicalGramSchmidt())
        α = norm(v)
    end
    v = scale!!(v, inv(α))

    r = apply_normal(operator, v)
    r = add!!(r, u, -α)
    β = norm(r)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(β)) < β < orth.η * nold
        nold = β
        r, = orthogonalize!!(r, U, ClassicalGramSchmidt())
        β = norm(r)
    end

    return v, r, α, β
end
function gklrecurrence(operator,
                       U::OrthonormalBasis,
                       V::OrthonormalBasis,
                       β,
                       orth::ModifiedGramSchmidtIR)
    u = U[end]
    v = apply_adjoint(operator, u)
    v = add!!(v, V[end], -β)
    α = norm(v)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(α)) < α < orth.η * nold
        nold = α
        for q in V
            v, = orthogonalize!!(v, q, ModifiedGramSchmidt())
        end
        α = norm(v)
    end
    v = scale!!(v, inv(α))

    r = apply_normal(operator, v)
    r = add!!(r, u, -α)
    β = norm(r)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(β)) < β < orth.η * nold
        nold = β
        for q in U
            r, = orthogonalize!!(r, q, ModifiedGramSchmidt())
        end
        β = norm(r)
    end

    return v, r, α, β
end
