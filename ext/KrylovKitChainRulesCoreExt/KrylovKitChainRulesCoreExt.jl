module KrylovKitChainRulesCoreExt

using KrylovKit
using ChainRulesCore
using LinearAlgebra
using VectorInterface

using KrylovKit: apply_normal, apply_adjoint
using KrylovKit: WARN_LEVEL, STARTSTOP_LEVEL, EACHITERATION_LEVEL

include("utilities.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("svdsolve.jl")

# mark some functions as non-differentiable to help Zygote:
@non_differentiable KrylovKit.apply_scalartype(args...)
@non_differentiable KrylovKit.genapply_scalartype(args...)

end # module
