module KrylovKit
# "vectors" should follow the following interface:
# length(v)::Int -> dimensionality of the vector space, i.e. number of elements in the vector
# eltype(v)::Type{<:Number} -> element type of the data in the vector
# similar(v, [T::Type{<:Number}]) -> a similar object, possibly with different eltype
# fill!(v, α) -> fill all entries of v with the value α, only used with α=0 to create a zero vector
# (we cannot use `scale!(v,v,0) because 0*NaN = NaN`)
# scale!(w,v,α) -> w = α*v: key properties of a vector
# LinAlg.axpy!(α,v,w) w += α*v : key properties of a vector
# vecdot(w,v), vecnorm(v) -> inner products and norms

using Compat

@static if !isdefined(Base, :AbstractRange)
    const IndexRange{T<:Integer} = Base.Range{T}
else
    const IndexRange{T<:Integer} = Base.AbstractRange{T}
end


using Base.length, Base.eltype, Base.similar, Base.LinAlg.axpy!, Base.scale!, Base.vecdot, Base.vecnorm

export orthogonalize, orthogonalize!, orthonormalize, orthonormalize!
export cgs, mgs, cgs2, mgs2, cgsr, mgsr
export rayleighquotient, normres, residual, basis
export factorize, LanczosIterator, ArnoldiIterator
export linsolve, GMRES
export schursolve, eigsolve, Lanczos, Arnoldi
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
include("dense/reflector.jl")

# include("dense/factorizations.jl")
# include("dense/utldiv.jl")
# include("dense/hschur.jl")

struct ConvergenceInfo{S,T}
    converged::Int # how many vectors have converged, 0 or 1 for linear systems, exponentiate, any integer for eigenvalue problems
    normres::S
    residual::T
    numiter::Int
    numops::Int
end

include("linsolve/gmres.jl")

include("matrixfun/exponentiate.jl")

include("eigsolve/eigsolve.jl")
include("eigsolve/lanczos.jl")
include("eigsolve/arnoldi.jl")

apply(A::AbstractMatrix, x::AbstractVector) = A*x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = A_mul_B!(y, A, x)
apply!(y, f, x) = copy!(y, f(x))

include("recursivevec.jl")
export RecursiveVec
include("innerproductvec.jl")
export InnerProductVec

end
