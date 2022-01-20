# gkl.jl
"""
    mutable struct GKLFactorization{T,S<:Real}

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
mutable struct GKLFactorization{T,S<:Real}
    k::Int # current Krylov dimension
    U::OrthonormalBasis{T} # basis of length k
    V::OrthonormalBasis{T} # basis of length k
    αs::Vector{S}
    βs::Vector{S}
    r::T
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
Base.eltype(::Type{<:GKLFactorization{<:Any,S}}) where {S} = S

# iteration for destructuring into components
Base.iterate(F::GKLFactorization) = (basis(F, :U), Val(:V))
Base.iterate(F::GKLFactorization, ::Val{:V}) = (basis(F, :V), Val(:rayleighquotient))
Base.iterate(F::GKLFactorization, ::Val{:rayleighquotient}) =
    (rayleighquotient(F), Val(:residual))
Base.iterate(F::GKLFactorization, ::Val{:residual}) = (residual(F), Val(:normres))
Base.iterate(F::GKLFactorization, ::Val{:normres}) =
    (normres(F), Val(:rayleighextension))
Base.iterate(F::GKLFactorization, ::Val{:rayleighextension}) =
    (rayleighextension(F), Val(:done))
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
rayleighquotient(F::GKLFactorization) =
    Bidiagonal(view(F.αs, 1:F.k), view(F.βs, 1:(F.k-1)), :L)
residual(F::GKLFactorization) = F.r
@inbounds normres(F::GKLFactorization) = F.βs[F.k]
rayleighextension(F::GKLFactorization) = SimpleBasisVector(F.k, F.k)

# GKL iteration for constructing the orthonormal basis of a Krylov subspace.
"""
    struct GKLIterator{F,T,O<:Orthogonalizer}
    GKLIterator(f, u₀, [orth::Orthogonalizer = KrylovDefaults.orth, keepvecs::Bool = true])

Iterator that takes a general linear map `f::F` and an initial vector `u₀::T` and generates
an expanding `GKLFactorization` thereof. In particular, `GKLIterator` implements the
[Golub-Kahan-Lanczos bidiagonalization procedure](http://www.netlib.org/utk/people/JackDongarra/etemplates/node198.html).
Note, however, that this implementation starts from a vector `u₀` in the codomain of the
linear map `f`, which will end up (after normalisation) as the first column of `U`.

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
struct GKLIterator{F,T,O<:Orthogonalizer}
    operator::F
    u₀::T
    orth::O
    keepvecs::Bool
    function GKLIterator{F,T,O}(
        operator::F,
        u₀::T,
        orth::O,
        keepvecs::Bool
    ) where {F,T,O<:Orthogonalizer}
        if !keepvecs && isa(orth, Reorthogonalizer)
            error("Cannot use reorthogonalization without keeping all Krylov vectors")
        end
        return new{F,T,O}(operator, u₀, orth, keepvecs)
    end
end
GKLIterator(
    operator::F,
    u₀::T,
    orth::O = KrylovDefaults.orth,
    keepvecs::Bool = true
) where {F,T,O<:Orthogonalizer} = GKLIterator{F,T,O}(operator, u₀, orth, keepvecs)
GKLIterator(
    A::AbstractMatrix,
    u₀::AbstractVector,
    orth::O = KrylovDefaults.orth,
    keepvecs::Bool = true
) where {O<:Orthogonalizer} =
    GKLIterator((x, flag) -> flag ? A' * x : A * x, u₀, orth, keepvecs)

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

function initialize(iter::GKLIterator; verbosity::Int = 0)
    # initialize without using eltype
    u₀ = iter.u₀
    β₀ = norm(u₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    v₀ = iter.operator(u₀, true) # apply adjoint operator, might change eltype
    α = norm(v₀) / β₀
    Av₀ = iter.operator(v₀, false) # apply operator
    α² = dot(u₀, Av₀) / β₀^2
    α² ≈ α * α || throw(ArgumentError("operator and its adjoint are not compatible"))
    T = typeof(α²)
    # this line determines the type that we will henceforth use
    u = (one(T) / β₀) * u₀ # u = mul!(similar(u₀, T), u₀, 1/β₀)
    if typeof(v₀) == typeof(u)
        v = rmul!(v₀, 1 / (α * β₀))
    else
        v = mul!(similar(u), v₀, 1 / (α * β₀))
    end
    if typeof(Av₀) == typeof(u)
        r = rmul!(Av₀, 1 / (α * β₀))
    else
        r = mul!(similar(u), Av₀, 1 / (α * β₀))
    end
    r = axpy!(-α, u, r)
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
function initialize!(iter::GKLIterator, state::GKLFactorization; verbosity::Int = 0)
    U = state.U
    while length(U) > 1
        pop!(U)
    end
    V = empty!(state.V)
    αs = empty!(state.αs)
    βs = empty!(state.βs)

    u = mul!(V[1], iter.u₀, 1 / norm(iter.u₀))
    v = iter.operator(u, true)
    α = norm(v)
    rmul!(v, 1 / α)
    r = iter.operator(v, false) # apply operator
    r = axpy!(-α, u, r)
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
function expand!(iter::GKLIterator, state::GKLFactorization; verbosity::Int = 0)
    βold = normres(state)
    U = state.U
    V = state.V
    r = state.r
    U = push!(U, rmul!(r, 1 / βold))
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
    state.r = rmul!(r, normres(state))
    return state
end

# Golub-Kahan-Lanczos recurrence relation
function gklrecurrence(
    operator,
    U::OrthonormalBasis,
    V::OrthonormalBasis,
    β,
    orth::Union{ClassicalGramSchmidt,ModifiedGramSchmidt}
)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(
    operator,
    U::OrthonormalBasis,
    V::OrthonormalBasis,
    β,
    orth::ClassicalGramSchmidt2
)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v) # not necessary if we definitely reorthogonalize next step and previous step
    # v, = orthogonalize!(v, V, ClassicalGramSchmidt())
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    r, = orthogonalize!(r, U, ClassicalGramSchmidt())
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(
    operator,
    U::OrthonormalBasis,
    V::OrthonormalBasis,
    β,
    orth::ModifiedGramSchmidt2
)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    # for q in V # not necessary if we definitely reorthogonalize next step and previous step
    #     v, = orthogonalize!(v, q, ModifiedGramSchmidt())
    # end
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    for q in U
        r, = orthogonalize!(r, q, ModifiedGramSchmidt())
    end
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(
    operator,
    U::OrthonormalBasis,
    V::OrthonormalBasis,
    β,
    orth::ClassicalGramSchmidtIR
)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    α = norm(v)
    nold = sqrt(abs2(α) + abs2(β))
    while α < orth.η * nold
        nold = α
        v, = orthogonalize!(v, V, ClassicalGramSchmidt())
        α = norm(v)
    end
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    β = norm(r)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(β)) < β < orth.η * nold
        nold = β
        r, = orthogonalize!(r, U, ClassicalGramSchmidt())
        β = norm(r)
    end

    return v, r, α, β
end
function gklrecurrence(
    operator,
    U::OrthonormalBasis,
    V::OrthonormalBasis,
    β,
    orth::ModifiedGramSchmidtIR
)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    α = norm(v)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(α)) < α < orth.η * nold
        nold = α
        for q in V
            v, = orthogonalize!(v, q, ModifiedGramSchmidt())
        end
        α = norm(v)
    end
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    β = norm(r)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(β)) < β < orth.η * nold
        nold = β
        for q in U
            r, = orthogonalize!(r, q, ModifiedGramSchmidt())
        end
        β = norm(r)
    end

    return v, r, α, β
end
