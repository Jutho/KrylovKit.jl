"""
    abstract type KrylovFactorization{T,S<:Number}

Abstract type to store a Krylov factorization of a linear map `A` of the form

```julia
A * V = V * B + r * b'
```

For a given Krylov factorization `fact` of length `k = length(fact)`, the basis `V` is
obtained via [`basis(fact)`](@ref basis) and is an instance of some subtype of
[`Basis{T}`](@ref Basis), with also `length(V) == k` and where `T` denotes the type of
vector like objects used in the problem. The Rayleigh quotient `B` is obtained as
[`rayleighquotient(fact)`](@ref) and `typeof(B)` is some subtype of `AbstractMatrix{S}` with
`size(B) == (k,k)`, typically a structured matrix. The residual `r` is obtained as
[`residual(fact)`](@ref) and is of type `T`. One can also query [`normres(fact)`](@ref) to
obtain `norm(r)`, the norm of the residual. The vector `b` has no dedicated name and often
takes a default form (see below). It should be a subtype of `AbstractVector` of length `k`
and can be obtained as [`rayleighextension(fact)`](@ref) (by lack of a better dedicated
name).

A Krylov factorization `fact` can be destructured as `V, B, r, nr, b = fact` with
`nr = norm(r)`.

See also [`LanczosFactorization`](@ref) and [`ArnoldiFactorization`](@ref) for concrete
implementations, and [`KrylovIterator`](@ref) (with in particular [`LanczosIterator`](@ref)
and [`ArnoldiIterator`](@ref)) for iterators that construct progressively expanding Krylov
factorizations of a given linear map and a starting vector.
"""
abstract type KrylovFactorization{T,S} end

"""
    abstract type KrylovIterator{F,T}

Abstract type for iterators that take a linear map of type `F` and an initial vector of type
`T` and generate an expanding `KrylovFactorization` thereof.

When iterating over an instance of `KrylovIterator`, the values being generated are subtypes
of [`KrylovFactorization`](@ref), which can be immediately destructured into a
[`basis`](@ref), [`rayleighquotient`](@ref), [`residual`](@ref), [`normres`](@ref) and
[`rayleighextension`](@ref).

See [`LanczosIterator`](@ref) and [`ArnoldiIterator`](@ref) for concrete implementations and
more information.
"""
abstract type KrylovIterator{F,T} end

"""
        basis(fact::KrylovFactorization)

Return the list of basis vectors of a [`KrylovFactorization`](@ref), which span the Krylov
subspace. The return type is a subtype of `Basis{T}`, where `T` represents the type of the
vectors used by the problem.
"""
function basis end

"""
    rayleighquotient(fact::KrylovFactorization)

Return the Rayleigh quotient of a [`KrylovFactorization`](@ref), i.e. the reduced matrix
within the basis of the Krylov subspace. The return type is a subtype of
`AbstractMatrix{<:Number}`, typically some structured matrix type.
"""
function rayleighquotient end

"""
    residual(fact::KrylovFactorization)

Return the residual of a [`KrylovFactorization`](@ref). The return type is some vector of
the same type as used in the problem. See also [`normres(F)`](@ref) for its norm, which
typically has been computed already.
"""
function residual end

"""
    normres(fact::KrylovFactorization)

Return the norm of the residual of a [`KrylovFactorization`](@ref). As this has typically
already been computed, it is cheaper than (but otherwise equivalent to) `norm(residual(F))`.
"""
function normres end

"""
    rayleighextension(fact::KrylovFactorization)

Return the vector `b` appearing in the definition of a [`KrylovFactorization`](@ref); often
it is simply the last coordinate unit vector, which can be represented using
[`SimpleBasisVector`](@ref).
"""
function rayleighextension end

"""
    shrink!(fact::KrylovFactorization, k)

Shrink an existing Krylov factorization `fact` down to have length `k`. Does nothing if
`length(fact)<=k`.
"""
function shrink! end

"""
    expand!(iter::KrylovIterator, fact::KrylovFactorization)

Expand the Krylov factorization `fact` by one using the linear map and parameters in `iter`.
"""
function expand! end

"""
    initialize!(iter::KrylovIterator, fact::KrylovFactorization)

Initialize a length 1 Krylov factorization corresponding to `iter` in the already existing
factorization `fact`, thereby destroying all the information it currently holds.
"""
function initialize! end

"""
    initialize(iter::KrylovIterator)

Initialize a length 1 Krylov factorization corresponding to `iter`.
"""
function initialize end

# iteration for destructuring into components
Base.iterate(F::KrylovFactorization) = (basis(F), Val(:rayleighquotient))
function Base.iterate(F::KrylovFactorization, ::Val{:rayleighquotient})
    return (rayleighquotient(F), Val(:residual))
end
Base.iterate(F::KrylovFactorization, ::Val{:residual}) = (residual(F), Val(:normres))
function Base.iterate(F::KrylovFactorization, ::Val{:normres})
    return (normres(F), Val(:rayleighextension))
end
function Base.iterate(F::KrylovFactorization, ::Val{:rayleighextension})
    return (rayleighextension(F), Val(:done))
end
Base.iterate(F::KrylovFactorization, ::Val{:done}) = nothing
