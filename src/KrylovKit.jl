module KrylovKit
# "vectors" should follow the following interface:
# length(v)::Int -> dimensionality of the vector space, i.e. number of elements in the vector
# eltype(typeof(v))::Type{<:Number} -> element type of the data in the vector
# similar(v, [T::Type{<:Number}]) -> a similar object, possibly with different eltype
# copyto!(w, v) -> copy vector v to pre-allocated vector w (e.g. created with similar, possibly with different eltype)
# fill!(v, α) -> fill all entries of v with the value α, only used with α=0 to create a zero vector
# (we cannot use `scale!(v,v,0) because 0*NaN = NaN`)
# mul!(w,v,α) -> w = α*v: key properties of a vector
# axpy!(α,v,w) w += α*v : key properties of a vector
# axpby!(α,v,β,w) w = α*v + β*w : key properties of a vector
# dot(w,v), norm(v) -> inner products and norms

using LinearAlgebra
const IndexRange = AbstractRange{Int64}

export orthogonalize, orthogonalize!, orthonormalize, orthonormalize!
export cgs, mgs, cgs2, mgs2, cgsr, mgsr
export rayleighquotient, normres, residual, basis
export factorize, initialize, expand!, shrink!
export LanczosIterator, ArnoldiIterator
export linsolve, GMRES
export schursolve, eigsolve, Lanczos, Arnoldi
export exponentiate
export ConvergenceInfo

include("algorithms.jl")

abstract type Basis{T} end
abstract type KrylovFactorization{T} end

include("orthonormal.jl")

include("dense/givens.jl")
include("dense/linalg.jl")
include("dense/packedhessenberg.jl")
include("dense/reflector.jl")

include("factorize/lanczos.jl")
include("factorize/arnoldi.jl")

#
# # include("dense/factorizations.jl")
# # include("dense/utldiv.jl")
# # include("dense/hschur.jl")
#
struct ConvergenceInfo{S,T}
    converged::Int # how many vectors have converged, 0 or 1 for linear systems, exponentiate, any integer for eigenvalue problems
    normres::S
    residual::T
    numiter::Int
    numops::Int
end

include("eigsolve/eigsolve.jl")
include("eigsolve/lanczos.jl")
include("eigsolve/arnoldi.jl")

include("linsolve/gmres.jl")

include("matrixfun/exponentiate.jl")

apply(A::AbstractMatrix, x::AbstractVector) = A*x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
apply!(y, f, x) = copyto!(y, f(x))

# include("recursivevec.jl")
# export RecursiveVec
# include("innerproductvec.jl")
# export InnerProductVec

end
