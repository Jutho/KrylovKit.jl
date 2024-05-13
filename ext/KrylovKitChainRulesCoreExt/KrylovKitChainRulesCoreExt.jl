module KrylovKitChainRulesCoreExt

using KrylovKit
using ChainRulesCore
using LinearAlgebra
using VectorInterface

include("linsolve.jl")
include("eigsolve.jl")

end # module
