module KrylovKit

import Base.LinAlg.axpy!, Base.vecdot, Base.vecnorm, Base.scale!


export cgs, mgs, cgs2, mgs2, cgsr, mgsr
export lanczos, arnoldi
export linsolve
export GMRES
export eigsolve, SimpleArnoldi, RestartedArnoldi

include("algorithms.jl")

abstract type Basis{T} end

include("orthonormal.jl")
include("factorize/lanczos.jl")
include("factorize/arnoldi.jl")

include("dense/givens.jl")
include("dense/reflector.jl")
include("dense/factorizations.jl")
include("dense/utldiv.jl")
include("dense/hschur.jl")

include("linsolve/gmres.jl")

include("eigsolve/eigsolve.jl")
include("eigsolve/arnoldi.jl")

struct ConvergenceInfo{S,T}
    converged::Int
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
