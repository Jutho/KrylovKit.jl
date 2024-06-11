module KrylovKitChainRulesCoreExt

using KrylovKit
using ChainRulesCore
using LinearAlgebra
using VectorInterface

using KrylovKit: apply_normal, apply_adjoint

include("utilities.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("svdsolve.jl")

end # module
