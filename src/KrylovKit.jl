"""
    KrylovKit

A Julia package collecting a number of Krylov-based algorithms for linear problems,
singular value and eigenvalue problems and the application of functions of linear maps or
operators to vectors.

KrylovKit accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   [`linsolve`](@ref): solve linear systems
*   [`eigsolve`](@ref): find a few eigenvalues and corresponding eigenvectors
*   [`geneigsolve`](@ref): find a few generalized eigenvalues and corresponding vectors
*   [`svdsolve`](@ref): find a few singular values and corresponding left and right
    singular vectors
*   [`exponentiate`](@ref): apply the exponential of a linear map to a vector
"""
module KrylovKit

using LinearAlgebra
using Printf
const IndexRange = AbstractRange{Int}

export linsolve, eigsolve, geneigsolve, svdsolve, schursolve, exponentiate, expintegrator
export orthogonalize, orthogonalize!, orthonormalize, orthonormalize!
export basis, rayleighquotient, residual, normres, rayleighextension
export initialize, initialize!, expand!, shrink!
export ClassicalGramSchmidt, ClassicalGramSchmidt2, ClassicalGramSchmidtIR
export ModifiedGramSchmidt, ModifiedGramSchmidt2, ModifiedGramSchmidtIR
export LanczosIterator, ArnoldiIterator, GKLIterator
export CG, GMRES, Lanczos, Arnoldi, GKL, GolubYe
export KrylovDefaults, ClosestTo, EigSorter
export RecursiveVec, InnerProductVec

include("algorithms.jl")

# Structures to store a list of basis vectors
"""
    abstract type Basis{T} end

An abstract type to collect specific types for representing a basis of vectors of type `T`.

Implementations of `Basis{T}` behave in many ways like `Vector{T}` and should have a
`length`, can be indexed (`getindex` and `setindex!`), iterated over (`iterate`), and
support resizing (`resize!`, `pop!`, `push!`, `empty!`, `sizehint!`).

The type `T` denotes the type of the elements stored in an `Basis{T}` and can be any custom type that has vector like behavior (as defined in the docs of [`KrylovKit`](@ref)).

See [`OrthonormalBasis`](@ref).
"""
abstract type Basis{T} end

include("orthonormal.jl")

# Dense linear algebra structures and functions used in the algorithms below
include("dense/givens.jl")
include("dense/linalg.jl")
include("dense/packedhessenberg.jl")
include("dense/reflector.jl")

# Simple coordinate basis vector, i.e. a vector of all zeros and a single one on position `k`:
"""
    SimpleBasisVector(m, k)

Construct a simple struct `SimpleBasisVector <: AbstractVector{Bool}` representing a
coordinate basis vector of length `m` in the direction of `k`, i.e. for `e_k =
SimpleBasisVector(m, k)` we have `length(e_k) = m` and `e_k[i] = (i == k)`.
"""
struct SimpleBasisVector <: AbstractVector{Bool}
    m::Int
    k::Int
end
Base.size(e::SimpleBasisVector) = (e.m,)
@inline function Base.getindex(e::SimpleBasisVector, i)
    @boundscheck Base.checkbounds(e, i)
    return e.k == i
end

# Krylov factorizations and their iterators, the central objects for writing algorithms in KrylovKit
abstract type KrylovFactorization{T,S} end
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

Return the vector `b` appearing in the definition of a [`KrylovFactorization`](@ref).
"""
function rayleighextension end

"""
    shrink!(fact::KrylovFactorization, k)

Shrink an existing Krylov factorization `fact` down to have length `k`. Does nothing if
`length(fact)<=k`.
"""
function shrink! end

"""
    expand!(iter::KrylovIteraotr, fact::KrylovFactorization)

Expand the Krylov factorization `fact` by one using the linear map and parameters in `iter`.
"""
function expand! end

"""
    initialize!(iter::KrylovIteraotr, fact::KrylovFactorization)

Initialize a length 1 Kryov factorization corresponding to `iter` in the already existing
factorization `fact`, thereby destroying all the information it currently holds.
"""
function initialize! end

"""
    initialize(iter::KrylovIteraotr)

Initialize a length 1 Kryov factorization corresponding to `iter`.
"""
function initialize end

# iteration for destructuring into components
Base.iterate(F::KrylovFactorization) = (basis(F), Val(:rayleighquotient))
Base.iterate(F::KrylovFactorization, ::Val{:rayleighquotient}) = (rayleighquotient(F), Val(:residual))
Base.iterate(F::KrylovFactorization, ::Val{:residual}) = (residual(F), Val(:normres))
Base.iterate(F::KrylovFactorization, ::Val{:normres}) = (normres(F), Val(:rayleighextension))
Base.iterate(F::KrylovFactorization, ::Val{:rayleighextension}) = (rayleighextension(F), Val(:done))
Base.iterate(F::KrylovFactorization, ::Val{:done}) = nothing

include("krylov/lanczos.jl")
include("krylov/arnoldi.jl")
include("krylov/gkl.jl")

"""
    abstract type KrylovFactorization{T,S<:Number}
    mutable struct LanczosFactorization{T,S<:Real}    <: KrylovFactorization{T,S}
    mutable struct ArnoldiFactorization{T,S<:Number}  <: KrylovFactorization{T,S}

Structures to store a Krylov factorization of a linear map `A` of the form
```julia
    A * V = V * B + r * b'.
```
For a given Krylov factorization `fact` of length `k = length(fact)`, the basis `A is
obtained via [`basis(fact)`](@ref basis) and is an instance of some subtype of
[`Basis{T}`](@ref Basis), with also `length(V) == k` and where `T` denotes the type of
vector like objects used in the problem. The Rayleigh quotient `B` is obtained as
[`rayleighquotient(fact)`](@ref) and `typeof(B)` is some subtype of `AbstractMatrix{S}`
with `size(B) == (k,k)`, typically a structured matrix. The residual `r` is obtained as
[`residual(fact)`](@ref) and is of type `T`. One can also query [`normres(fact)`](@ref) to
obtain `norm(r)`, the norm of the residual. The vector `b` has no dedicated name and often
takes a default form (see below). It should be a subtype of `AbstractVector` of length `k`
and can be obtained as [`rayleighextension(fact)`](@ref) (by lack of a better dedicated
name).

In particular, `LanczosFactorization` stores a Lanczos factorization of a real symmetric or
complex hermitian linear map and has `V::OrthonormalBasis{T}` and
`B::SymTridiagonal{S<:Real}`. `ArnoldiFactorization` stores an Arnoldi factorization of a
general linear map and has `V::OrthonormalBasis{T}` and
[`B::PackedHessenberg{S<:Number}`](@ref PackedHessenberg). In both
cases, `b` takes the default value ``e_k``, i.e. the unit vector of all zeros and a one in
the last entry, which is represented using [`SimpleBasisVector`](@ref).

A Krylov factorization `fact` can be destructured as `V, B, r, nr, b = fact` with `nr = norm(r)`.

`LanczosFactorization` and `ArnoldiFactorization` are mutable because they can
[`expand!`](@ref) or [`shrink!`](@ref). See also [`KrylovIterator`](@ref) (and in
particular [`LanczosIterator`](@ref) and [`ArnoldiIterator`](@ref)) for iterators that
construct progressively expanding Krylov factorizations of a given linear map and a
starting vector.
"""
KrylovFactorization, LanczosFactorization, ArnoldiFactorization

"""
    abstract type KrylovIterator{F,T}
    struct LanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    struct ArnoldiIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}

    LanczosIterator(f, v₀, [orth::Orthogonalizer = KrylovDefaults.orth], keepvecs::Bool = true)
    ArnoldiIterator(f, v₀, [orth::Orthogonalizer = KrylovDefaults.orth])

Iterators that take a linear map of type `F` and an initial vector of type `T` and generate
an expanding `KrylovFactorization` thereof.

In particular, for a real symmetric or complex hermitian linear map `f`, `LanczosIterator`
uses the [Lanczos iteration](https://en.wikipedia.org/wiki/Lanczos_algorithm) scheme to
build a successively expanding `LanczosFactorization`. While `f` cannot be tested to be
symmetric or hermitian directly when the linear map is encoded as a general callable object
or function, it is tested whether the imaginary part of `dot(v, f(v))` is sufficiently
small to be neglected.

Similarly, for a general linear map `f`, `ArnoldiIterator` iterates over progressively
expanding `ArnoldiFactorizations` using the [Arnoldi
iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration).

The optional argument `orth` specifies which [`Orthogonalizer`](@ref) to be used. The
default value in [`KrylovDefaults`](@ref) is to use [`ModifiedGramSchmidtIR`](@ref), which
possibly uses reorthogonalization steps. For `LanczosIterator`, one can use to discard the
old vectors that span the Krylov subspace by setting the final argument `keepvecs` to
`false`. This, however, is only possible if an `orth` algorithm is used that does not rely
on reorthogonalization, such as `ClassicalGramSchmidt()` or `ModifiedGramSchmidt()`. In
that case, the iterator strictly uses the Lanczos three-term recurrence relation.

When iterating over an instance of `KrylovIterator`, the values being generated are subtypes
of [`KrylovFactorization`](@ref), which can be immediately destructured into a
[`basis`](@ref), [`rayleighquotient`](@ref), [`residual`](@ref), [`normres`](@ref) and
[`rayleighextension`](@ref), for example as
```julia
for V,B,r,nr,b in ArnoldiIterator(f, v₀)
    # do something
    nr < tol && break # a typical stopping criterion
end
```
Note, however, that if `keepvecs=false` in `LanczosIterator`, the basis `V` cannot be
extracted. Since the iterators don't know the dimension of the underlying vector space of
objects of type `T`, they keep expanding the Krylov subspace until `normres` falls below
machine precision `eps` for the given `eltype(T)`.

The internal state of `LanczosIterator` and `ArnoldiIterator` is the same as the return
value, i.e. the corresponding `LanczosFactorization` or `ArnoldiFactorization`. However, as
Julia's Base iteration interface (using `Base.iterate`) requires that the state is not
mutated, a `deepcopy` is produced upon every next iteration step.

Instead, you can also mutate the `KrylovFactorization` in place, using the following
interface, e.g. for the same example above
```julia
iterator = ArnoldiIterator(f, v₀)
factorization = initialize(iterator)
while normres(factorization) > tol
    expand!(iterator, f)
    V,B,r,nr,b = factorization
    # do something
end
```
Here, [`initialize(::KrylovIterator)`](@ref) produces the first Krylov factorization of
length 1, and `expand!(::KrylovIterator,::KrylovFactorization)`(@ref) expands the
factorization in place. See also
[`initialize!(::KrylovIterator,::KrylovFactorization)`](@ref) to initialize in an already
existing factorization (most information will be discarded) and
[`shrink!(::KrylovFactorization, k)`](@ref) to shrink an existing factorization down to
length `k`.
"""
KrylovIterator, LanczosIterator, ArnoldiIterator

# A general structure to pass on convergence information
"""
    struct ConvergenceInfo{S,T}
        converged::Int
        residual::T
        normres::S
        numiter::Int
        numops::Int
    end

Used to return information about the solution found by the iterative method.
*   `converged`: the number of solutions that have converged according to an appropriate error
    measure and requested tolerance for the problem. Its value can be zero or one for
    [`linsolve`](@ref) and [`exponentiate`](@ref), or any integer `>= 0` for
    [`eigsolve`](@ref), [`schursolve`](@ref) or [`svdsolve`]().
*   `residual:` the (list of) residual(s) for the problem, or `nothing` for problems without
    the concept of a residual (i.e. `exponentiate`). This is a single vector (of the same
    type as the type of vectors used in the problem) for `linsolve`, or a `Vector` of such
    vectors for `eigsolve`, `schursolve` or `svdsolve`.
*   `normres`: the norm of the residual(s) (in the previous field) or the value of any other
    error measure that is appropriate for the problem. This is a `Real` for `linsolve` and
    `exponentiate`, and a `Vector{<:Real}` for `eigsolve`, `schursolve` and `svdsolve`. The
    number of values in `normres` that are smaller than a predefined tolerance corresponds
    to the number `converged` of solutions that have converged.
*   `numiter`: the number of iterations (sometimes called restarts) used by the algorithm.
*   `numops`: the number of times the linear map or operator was applied
"""
struct ConvergenceInfo{S,T}
    converged::Int # how many vectors have converged, 0 or 1 for linear systems, exponentiate, any integer for eigenvalue problems
    residual::T
    normres::S
    numiter::Int
    numops::Int
end

function Base.show(io::IO, info::ConvergenceInfo)
    print(io, "ConvergenceInfo: ")
    info.converged == 0 && print(io, "no converged values ")
    info.converged == 1 && print(io, "one converged value ")
    info.converged > 1 && print(io, "$(info.converged) converged values ")
    println(io, "after $(info.numiter) iterations and $(info.numops) applications of the linear map;")
    println(io, "norms of residuals are given by $((info.normres...,)).")
end

# eigsolve en schursolve
include("eigsolve/eigsolve.jl")
include("eigsolve/lanczos.jl")
include("eigsolve/arnoldi.jl")
include("eigsolve/geneigsolve.jl")
include("eigsolve/golubye.jl")
include("eigsolve/svdsolve.jl")

# linsolve
include("linsolve/linsolve.jl")
include("linsolve/cg.jl")
include("linsolve/gmres.jl")

# exponentiate
include("matrixfun/exponentiate.jl")
include("matrixfun/expintegrator.jl")

apply(A::AbstractMatrix, x::AbstractVector) = A*x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
apply!(y, f, x) = copyto!(y, f(x))

include("recursivevec.jl")
include("innerproductvec.jl")

end
