"""
    KrylovKit

A Julia package collecting a number of Krylov-based algorithms for linear problems,
singular value and eigenvalue problems and the application of functions of linear maps or
operators to vectors.

KrylovKit accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior as vectors.

The high level interface of KrylovKit is provided by the following functions:

  - [`linsolve`](@ref): solve linear systems
  - [`eigsolve`](@ref): find a few eigenvalues and corresponding eigenvectors
  - [`geneigsolve`](@ref): find a few generalized eigenvalues and corresponding vectors
  - [`svdsolve`](@ref): find a few singular values and corresponding left and right
    singular vectors
  - [`exponentiate`](@ref): apply the exponential of a linear map to a vector
  - [`expintegrator`](@ref): exponential integrator for a linear non-homogeneous ODE,
    computes a linear combination of the `ϕⱼ` functions which generalize `ϕ₀(z) = exp(z)`.
"""
module KrylovKit
using VectorInterface
using VectorInterface: add!!
using LinearAlgebra
using Printf
using Random
using PackageExtensionCompat
const IndexRange = AbstractRange{Int}

export linsolve, reallinsolve
export eigsolve, geneigsolve, realeigsolve, schursolve, svdsolve
export exponentiate, expintegrator
export orthogonalize, orthogonalize!!, orthonormalize, orthonormalize!!
export basis, rayleighquotient, residual, normres, rayleighextension
export initialize, initialize!, expand!, shrink!
export ClassicalGramSchmidt, ClassicalGramSchmidt2, ClassicalGramSchmidtIR
export ModifiedGramSchmidt, ModifiedGramSchmidt2, ModifiedGramSchmidtIR
export LanczosIterator, ArnoldiIterator, GKLIterator
export CG, GMRES, BiCGStab, Lanczos, Arnoldi, GKL, GolubYe
export KrylovDefaults, EigSorter
export RecursiveVec, InnerProductVec

# Multithreading
const _NTHREADS = Ref(1)
get_num_threads() = _NTHREADS[]

function set_num_threads(n::Int)
    N = Base.Threads.nthreads()
    if n > N
        n = N
        _set_num_threads_warn(n)
    end
    return _NTHREADS[] = n
end
@noinline function _set_num_threads_warn(n)
    @warn "Maximal number of threads limited by number of Julia threads,
            setting number of threads equal to Threads.nthreads() = $n"
end

enable_threads() = set_num_threads(Base.Threads.nthreads())
disable_threads() = set_num_threads(1)

function __init__()
    @require_extensions
    set_num_threads(Base.Threads.nthreads())
    return nothing
end

struct SplitRange
    start::Int
    step::Int
    stop::Int
    innerlength::Int
    outerlength1::Int
    outerlength::Int
end
function splitrange(r::OrdinalRange, n::Integer)
    start = first(r)
    stp = step(r)
    stop = last(r)
    l = length(r)
    innerlength = div(l, n)
    outerlength1 = l - n * innerlength
    outerlength = n
    return SplitRange(start, stp, stop, innerlength, outerlength1, outerlength)
end
function Base.iterate(r::SplitRange, i=1)
    step = r.step
    if i <= r.outerlength1
        offset = (i - 1) * (r.innerlength + 1) * step
        start = r.start + offset
        stop = start + step * r.innerlength
    elseif i <= r.outerlength
        offset = (r.outerlength1 + (i - 1) * r.innerlength) * step
        start = r.start + offset
        stop = start + step * (r.innerlength - 1)
    else
        return nothing
    end
    return StepRange(start, step, stop), i + 1
end
Base.length(r::SplitRange) = r.outerlength

# Algorithm types
include("algorithms.jl")

# Structures to store a list of basis vectors
"""
    abstract type Basis{T} end

An abstract type to collect specific types for representing a basis of vectors of type `T`.

Implementations of `Basis{T}` behave in many ways like `Vector{T}` and should have a
`length`, can be indexed (`getindex` and `setindex!`), iterated over (`iterate`), and
support resizing (`resize!`, `pop!`, `push!`, `empty!`, `sizehint!`).

The type `T` denotes the type of the elements stored in an `Basis{T}` and can be any custom
type that has vector like behavior (as defined in the docs of KrylovKit).

See [`OrthonormalBasis`](@ref) for a specific implementation.
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
coordinate basis vector of length `m` in the direction of `k`, i.e. for
`e_k = SimpleBasisVector(m, k)` we have `length(e_k) = m` and `e_k[i] = (i == k)`.
"""
struct SimpleBasisVector <: AbstractVector{Bool}
    m::Int
    k::Int
end
Base.axes(e::SimpleBasisVector) = (Base.OneTo(e.m),)
Base.size(e::SimpleBasisVector) = (e.m,)
@inline function Base.getindex(e::SimpleBasisVector, i)
    @boundscheck Base.checkbounds(e, i)
    return e.k == i
end

# some often used tools
function checkposdef(z)
    r = checkhermitian(z)
    r > 0 || error("operator does not appear to be positive definite: diagonal element $z")
    return r
end
function checkhermitian(z, n=abs(z))
    imag(z) <= sqrt(max(eps(n), eps(one(n)))) ||
        error("operator does not appear to be hermitian: diagonal element $z")
    return real(z)
end

# apply operators
include("apply.jl")

# Krylov and related factorizations and their iterators
include("factorizations/krylov.jl")
include("factorizations/lanczos.jl")
include("factorizations/arnoldi.jl")
include("factorizations/gkl.jl")

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

  - `converged`: the number of solutions that have converged according to an appropriate
    error measure and requested tolerance for the problem. Its value can be zero or one for
    [`linsolve`](@ref), [`exponentiate`](@ref) and  [`expintegrator`](@ref), or any integer
    `>= 0` for [`eigsolve`](@ref), [`schursolve`](@ref) or [`svdsolve`](@ref).
  - `residual:` the (list of) residual(s) for the problem, or `nothing` for problems without
    the concept of a residual (i.e. `exponentiate`, `expintegrator`). This is a single
    vector (of the same type as the type of vectors used in the problem) for `linsolve`, or
    a `Vector` of such vectors for `eigsolve`, `schursolve` or `svdsolve`.
  - `normres`: the norm of the residual(s) (in the previous field) or the value of any other
    error measure that is appropriate for the problem. This is a `Real` for `linsolve` and
    `exponentiate`, and a `Vector{<:Real}` for `eigsolve`, `schursolve` and `svdsolve`. The
    number of values in `normres` that are smaller than a predefined tolerance corresponds
    to the number `converged` of solutions that have converged.
  - `numiter`: the number of iterations (sometimes called restarts) used by the algorithm.
  - `numops`: the number of times the linear map or operator was applied
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
    println(io,
            "after ",
            info.numiter,
            " iterations and ",
            info.numops,
            " applications of the linear map;")
    return println(io, "norms of residuals are given by $((info.normres...,)).")
end

# vectors with modified inner product
include("innerproductvec.jl")

# support for real
_realinner(v, w) = real(inner(v, w))
const RealVec{V} = InnerProductVec{typeof(_realinner),V}
RealVec(v) = InnerProductVec(v, _realinner)

apply(A, x::RealVec) = RealVec(apply(A, x[]))

# linsolve
include("linsolve/linsolve.jl")
include("linsolve/cg.jl")
include("linsolve/gmres.jl")
include("linsolve/bicgstab.jl")

# eigsolve and svdsolve
include("eigsolve/eigsolve.jl")
include("eigsolve/lanczos.jl")
include("eigsolve/arnoldi.jl")
include("eigsolve/geneigsolve.jl")
include("eigsolve/golubye.jl")
include("eigsolve/svdsolve.jl")

# exponentiate
include("matrixfun/exponentiate.jl")
include("matrixfun/expintegrator.jl")

# deprecations
include("deprecated.jl")

end
