module KrylovKit
# "vectors" should follow the following interface:
# length(v)::Int -> dimensionality of the vector space, i.e. number of elements in the vector
# eltype(v)::Type{<:Number} -> element type of the data in the vector
# similar(v, [T::Type{<:Number}]) -> a similar object, possibly with different eltype
# fill!(v, α) -> fill all entries of v with the value α
# scale!(w,v,α) -> w = α*v, LinAlg.axpy!(α,v,w) w += α*v : key properties of a vector
# vecdot(w,v), vecnorm(v) -> inner products and norms


using Base.length, Base.eltype, Base.similar, Base.LinAlg.axpy!, Base.scale!, Base.vecdot, Base.vecnorm

export cgs, mgs, cgs2, mgs2, cgsr, mgsr
export matrix, normres, residual, basis
export Lanczos, Arnoldi, NoRestart, ExplicitRestart, ImplicitRestart
export LanczosIterator, ArnoldiIterator
export linsolve, GMRES
export eigsolve
export exponentiate
export ConvergenceInfo

include("algorithms.jl")

abstract type Basis{T} end
abstract type KrylovFactorization{T} end

include("orthonormal.jl")
include("factorize/lanczos.jl")
include("factorize/arnoldi.jl")

include("dense/givens.jl")
include("dense/linalg.jl")
include("dense/packedhessenberg.jl")

# include("dense/factorizations.jl")
# include("dense/utldiv.jl")
# include("dense/hschur.jl")

include("linsolve/gmres.jl")
include("matrixfun/exponentiate.jl")

include("eigsolve/eigsolve.jl")
include("eigsolve/arnoldi.jl")

struct ConvergenceInfo{S,T}
    converged::Int # how many eigenvalues have converged, 0 or 1 for linear systems, ...
    normres::S
    residual::T
    numiter::Int
    numops::Int
end

apply(A::AbstractMatrix, x::AbstractVector) = A*x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = A_mul_B!(y, A, x)
apply!(y, f, x) = copy!(y, f(x))

end
