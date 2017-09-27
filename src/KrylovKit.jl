module KrylovKit

import Base.LinAlg.axpy!, Base.vecdot, Base.vecnorm, Base.scale!

export cgs, mgs, cgs2, mgs2, cgsr, mgsr
export matrix, normres, residual, basis
export LanczosIterator, ArnoldiIterator
export linsolve, GMRES
export eigsolve, Lanczos, Arnoldi, NoRestart, ExplicitRestart, ImplicitRestart
export ConvergenceInfo

include("algorithms.jl")

abstract type Basis{T} end
abstract type KrylovFactorization{T} end

include("orthonormal.jl")
include("factorize/lanczos.jl")
include("factorize/arnoldi.jl")

include("dense/givens.jl")
include("dense/linalg.jl")
# include("dense/factorizations.jl")
# include("dense/utldiv.jl")
# include("dense/hschur.jl")

include("linsolve/gmres.jl")

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

# include("schur.jl")

# include("factorize.jl")

# include("eigsolve.jl")



#
# module Defaults
# end
#
#
# export factorize, eigsolve
# export Arnoldi, RestartedArnoldi

end
