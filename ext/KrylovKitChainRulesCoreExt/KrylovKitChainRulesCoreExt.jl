module KrylovKitChainRulesCoreExt

using KrylovKit
using ChainRulesCore
using LinearAlgebra
using VectorInterface

include("svdsolve.jl")
include("linsolve.jl")
include("eigsolve.jl")

end # module
