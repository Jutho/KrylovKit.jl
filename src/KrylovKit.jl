"""
    KrylovKit

A package collecting a number of Krylov-based algorithms for linear problems, eigenvalue problems
and the application of functions of linear maps or operators to vectors.

So far, the starting point into KrylovKit are the general purpose functions [`linsolve`](ref),
[`eigsolve`](@ref), [`svdsolve`](@ref) and [`exponentiate`](@ref). KrylovKit distinguishes itself
from similar packages in the Julia ecosystem in the following two ways:

1.  `KrylovKit` accepts general functions `f` to represent the linear map or operator that defines
    the problem, but does of course also accept subtypes of `AbstractMatrix`. The linear map
    is always the first argument in `linsolve(f,...)`, `eigsolve(f,...)`, `exponentiate(f,...)`
    so that Julia's `do` block construction can be used, e.g.
    ```julia
    linsolve(...) do x
        # some linear operation on x
    end
    ```
    If the first argument `f isa AbstractMatrix`, matrix vector multiplication is used, otherwise
    `f` is called as a function.

2.  `KrylovKit` does not assume that the vectors involved in the problem are actual subtypes of
    `AbstractVector`. Any Julia object that behaves as a vector (in the way defined below) is
    supported, so in particular higher-dimensional arrays or any custom user type that supports
    the following functions (with `v` and `w` two instances of this type and `α` a scalar (`Number`)):
    *   `Base.eltype(v)`: the scalar type (i.e. `<:Number`) of the data in `v`
    *   `Base.similar(v, [T::Type<:Number])`: a way to construct additional similar vectors,
        possibly with a different scalar type `T`.
    *   `Base.copyto!(w, v)`: copy the contents of `v` to a preallocated vector `w`
    *   `Base.fill!(w, α)`: fill all the scalar entries of `w` with value `α`; this is only
        used in combination with `α = 0` to create a zero vector. Note that `Base.zero(v)` does
        not work for this purpose if we want to change the scalar `eltype`. We can also not
        use `rmul!(v, 0)` (see below), since `NaN*0` yields `NaN`.
    *   `LinearAlgebra.mul!(w, v, α)`: out of place scalar multiplication; multiply
        vector `v` with scalar `α` and store the result in `w`
    *   `LinearAlgebra.rmul!(v, α)`: in-place scalar multiplication of `v` with `α`.
    *   `LinearAlgebra.axpy!(α, v, w)`: store in `w` the result of `α*v + w`
    *   `LinearAlgebra.axpby!(α, v, β, w)`: store in `w` the result of `α*v + β*w`
    *   `LinearAlgebra.dot(v,w)`: compute the inner product of two vectors
    *   `LinearAlgebra.norm(v)`: compute the 2-norm of a vector

Currently implemented are the following algorithms:
*   `orthogonalization`: Classical & Modified Gram Schmidt, possibly with a second round or
    an adaptive number of rounds of reorthogonalization.
*   `linsolve`: [`GMRES`](@ref) without preconditioning
*   `eigsolve`: a Krylov-Schur algorithm (i.e. with tick restarts) for extremal eigenvalues of
    normal (i.e. not generalized) eigenvalue problems, corresponding to [`Lanczos`](@ref) for real symmetric
    or complex hermitian linear maps, and to [`Arnoldi`](@ref) for general linear maps.
*   `exponentiate`: a [`Lanczos`](@ref) based algorithm for the action of the exponential of a real
    symmetric or complex hermitian linear map.

Furthermore, `KrylovKit` provides two vector like types that might prove useful in certain applications.
*   [`RecursiveVec`](@ref) can be used for grouping a set of vectors into a single vector like
    structure (can be used recursively). The reason that e.g. `Vector{<:Vector}` cannot be used
    for this is that it returns the wrong `eltype` and methods like `similar(v, T)` and `fill!(v, α)`
    don't work correctly.
*   [`InnerProductVec`](@ref) can be used to redefine the inner product (i.e. `dot`) and corresponding
    norm (`norm`) of an already existing vector like object. The latter should help with implementing
    certain type of preconditioners and solving generalized eigenvalue problems with a positive
    definite matrix in the right hand side.

!!! note "A TODO list for the future"
    *   More linear solvers: conjugate gradient
    *   Biorthogonal methods for linear problems and eigenvalue problems: BiCG, BiCGStab, BiLanczos
    *   Generalized eigenvalue problems: LOPCG, EIGFP and trace minimization
    *   Exponentiate for a general (non-hermitian) linear map, based on `Arnoldi`
    *   Preconditioners
    *   Harmonic ritz values
    *   Nonlinear eigenvalue problems
    *   Support both in-place / mutating and out-of-place functions as linear maps
    *   Reuse memory for storing vectors when restarting algorithms
    *   Improved efficiency for the specific case where `x` is `Vector` (i.e. BLAS level 2 operations)
    *   More relevant matrix functions
"""
module KrylovKit

using LinearAlgebra
const IndexRange = AbstractRange{Int64}

export linsolve, eigsolve, svdsolve, schursolve, exponentiate
export orthogonalize, orthogonalize!, orthonormalize, orthonormalize!
export basis, rayleighquotient, residual, normres, rayleighextension
export initialize, initialize!, expand!, shrink!
export ClassicalGramSchmidt, ClassicalGramSchmidt2, ClassicalGramSchmidtIR
export ModifiedGramSchmidt, ModifiedGramSchmidt2, ModifiedGramSchmidtIR
export LanczosIterator, ArnoldiIterator
export GMRES, Lanczos, Arnoldi
export KrylovDefaults, ClosestTo
export RecursiveVec, InnerProductVec

include("algorithms.jl")

# Structures to store a list of basis vectors
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

Construct a simple struct `SimpleBasisVector <: AbstractVector{Bool}` representing a coordinate
basis vector of length `m` in the direction of `k`, i.e. for `e_k = SimpleBasisVector(m, k)`
we have `length(e_k) = m` and `e_k[i] = (i == k)`.
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

Return the list of basis vectors of a [`KrylovFactorization`](@ref), which span the Krylov subspace.
The return type is a subtype of `Basis{T}`, where `T` represents the type of the vectors used
by the problem.
"""
function basis end

"""
    rayleighquotient(fact::KrylovFactorization)

Return the Rayleigh quotient of a [`KrylovFactorization`](@ref), i.e. the reduced matrix within
the basis of the Krylov subspace. The return type is a subtype of `AbstractMatrix{<:Number}`,
typically some structured matrix type.
"""
function rayleighquotient end

"""
    residual(fact::KrylovFactorization)

Return the residual of a [`KrylovFactorization`](@ref). The return type is some vector of the
same type as used in the problem. See also [`normres(F)`](@ref) for its norm, which typically
has been computed already.
"""
function residual end

"""
    normres(fact::KrylovFactorization)

Return the norm of the residual of a [`KrylovFactorization`](@ref). As this has typically already
been computed, it is cheaper than (but otherwise equivalent to) `norm(residual(F))`.
"""
function normres end

"""
    rayleighextension(fact::KrylovFactorization)

Return the vector `b` appearing in the definition of a [`KrylovFactorization`](@ref).
"""
function rayleighextension end

"""
    shrink!(fact::KrylovFactorization, k)

Shrink an existing Krylov factorization `fact` down to have length `k`. Does nothing if `length(fact)<=k`.
"""
function shrink! end

"""
    expand!(iter::KrylovIteraotr, fact::KrylovFactorization)

Expand the Krylov factorization `fact` by one using the linear map and parameters in `iter`.
"""
function expand! end

"""
    initialize!(iter::KrylovIteraotr, fact::KrylovFactorization)

Initialize a length 1 Kryov factorization corresponding to `iter` in the already existing factorization
`fact`, thereby destroying all the information it currently holds.
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

include("factorize/lanczos.jl")
include("factorize/arnoldi.jl")

"""
    abstract type KrylovFactorization{T,S<:Number}
    mutable struct LanczosFactorization{T,S<:Real}    <: KrylovFactorization{T,S}
    mutable struct ArnoldiFactorization{T,S<:Number}  <: KrylovFactorization{T,S}

Structures to store a Krylov factorization of a linear map `A` of the form
```math
    A * V = V * B + r * b'.
```
For a given Krylov factorization `fact` of length `k = length(fact)`, the basis ``V`` is obtained
via [`basis(fact)`](@ref basis) and is an instance of some subtype of [`Basis{T}`](@ref Basis),
with also `length(V) == k` and where `T` denotes the type of vector like objects used in the
problem. The Rayleigh quotient ``B`` is obtained as [`rayleighquotient(fact)`](@ref) and `typeof(B)`
is some subtype of `AbstractMatrix{S}` with `size(B) == (k,k)`, typically a structured matrix.
The residual `r` is obtained as [`residual(fact)`](@ref) and is of type `T`. One can also query
[`normres(fact)`](@ref) to obtain `norm(r)`, the norm of the residual. The vector ``b`` has no
dedicated name and often takes a default form (see below). It should be a subtype of `AbstractVector`
of length `k` and can be obtained as [`rayleighextension(fact)`](@ref) (by lack of a better dedicated name).

In particular, `LanczosFactorization` stores a Lanczos factorization of a real symmetric or
complex hermitian linear map and has `V::OrthonormalBasis{T}` and `B::SymTridiagonal{S<:Real}`.
`ArnoldiFactorization` stores an Arnoldi factorization of a general linear map and has
`V::OrthonormalBasis{T}` and [`B::PackedHessenberg{S<:Number}`](@ref PackedHessenberg). In both
cases, ``b`` takes the default value ``e_k``, i.e. the unit vector of all zeros and a one in
the last entry, which is represented using [`SimpleBasisVector`](@ref).

A Krylov factorization `fact` can be destructured as `V, B, r, nr, b = fact` with `nr = norm(r)`.

`LanczosFactorization` and `ArnoldiFactorization` are mutable because they can [`expand!`](@ref)
or [`shrink!`](@ref). See also [`KrylovIterator`](@ref) (and in particular [`LanczosIterator`](@ref)
and [`ArnoldiIterator`](@ref)) for iterators that construct progressively expanding Krylov factorizations
of a given linear map and a starting vector.
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

In particular, for a real symmetric or complex hermitian linear map `f`, `LanczosIterator` uses
the [Lanczos iteration](https://en.wikipedia.org/wiki/Lanczos_algorithm) scheme to build a successively
expanding `LanczosFactorization`. While `f` cannot be tested to be symmetric or hermitian directly
when the linear map is encoded as a general callable object or function, it is tested whether
the imaginary part of `dot(v, f(v))` is sufficiently small to be neglected.

Similarly, for a general linear map `f`, `ArnoldiIterator` iterates over progressively expanding
`ArnoldiFactorizations` using the [Arnoldi iteration](https://en.wikipedia.org/wiki/Arnoldi_iteration).

The optional argument `orth` specifies which [`Orthogonalizer`](@ref) to be used. The default
value in [`KrylovDefaults`](@ref) is to use [`ModifiedGramSchmidtIR`](@ref), which possibly uses
reorthogonalization steps. For `LanczosIterator`, one can use to discard the old vectors that
span the Krylov subspace by setting the final argument `keepvecs` to `false`. This, however, is
only possible if an `orth` algorithm is used that does not rely on reorthogonalization, such as
`ClassicalGramSchmidt()` or `ModifiedGramSchmidt()`. In that case, the iterator strictly uses
the Lanczos three-term recurrence relation.

When iterating over an instance of `KrylovIterator`, the values being generated are subtypes
of [`KrylovFactorization`](@ref), which can be immediately destructured into a [`basis`](@ref),
[`rayleighquotient`](@ref), [`residual`](@ref), [`normres`](@ref) and [`rayleighextension`](@ref),
for example as
```julia
for V,B,r,nr,b in ArnoldiIterator(f, v₀)
    # do something
    nr < tol && break # a typical stopping criterion
end
```
Note, however, that if `keepvecs=false` in `LanczosIterator`, the basis `V` cannot be extracted.
Since the iterators don't know the dimension of the underlying vector space of objects of type `T`,
they keep expanding the Krylov subspace until `normres` falls below machine precision `eps`
for the given `eltype(T)`.

The internal state of `LanczosIterator` and `ArnoldiIterator` is the same as the return value,
i.e. the corresponding `LanczosFactorization` or `ArnoldiFactorization`. However, as Julia's
Base iteration interface (using `Base.iterate`) requires that the state is not mutated, a `deepcopy`
is produced upon every next iteration step.

Instead, you can also mutate the `KrylovFactorization` in place, using the following interface,
e.g. for the same example above
```julia
iterator = ArnoldiIterator(f, v₀)
factorization = initialize(iterator)
while normres(factorization) > tol
    expand!(iterator, f)
    V,B,r,nr,b = factorization
    # do something
end
```
Here, [`initialize(::KrylovIterator)`](@ref) produces the first Krylov factorization of length
1, and `expand!(::KrylovIterator,::KrylovFactorization)`(@ref) expands the factorization in place.
See also [`initialize!(::KrylovIterator,::KrylovFactorization)`](@ref) to initialize in an already
existing factorization (most information will be discarded) and [`shrink!(::KrylovFactorization, k)`](@ref)
to shrink an existing factorization down to length `k`.
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
    measure and requested tolerance for the problem. Its value can be zero or one for [`linsolve`](@ref)
    and [`exponentiate`](@ref), or any integer `>= 0` for [`eigsolve`](@ref), [`schursolve`](@ref)
    or [`svdsolve`]().
*   `residual:` the (list of) residual(s) for the problem, or `nothing` for problems without
    the concept of a residual (i.e. `exponentiate`). This is a single vector (of the same type
    as the type of vectors used in the problem) for `linsolve`, or a `Vector` of such vectors
    for `eigsolve`, `schursolve` or `svdsolve`.
*   `normres`: the norm of the residual(s) (in the previous field) or the value of any other
    error measure that is appropriate for the problem. This is a `Real` for `linsolve` and
    `exponentiate`, and a `Vector{<:Real}` for `eigsolve`, `schursolve` and `svdsolve`. The
    number of values in `normres` that are smaller than a predefined tolerance corresponds to
    the number `converged` of solutions that have converged.
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

# eigsolve en schursolve
include("eigsolve/eigsolve.jl")
include("eigsolve/lanczos.jl")
include("eigsolve/arnoldi.jl")
include("eigsolve/svdsolve.jl")

# linsolve
include("linsolve/linsolve.jl")
include("linsolve/gmres.jl")

# exponentiate
include("matrixfun/exponentiate.jl")

apply(A::AbstractMatrix, x::AbstractVector) = A*x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
apply!(y, f, x) = copyto!(y, f(x))

include("recursivevec.jl")
include("innerproductvec.jl")

end
