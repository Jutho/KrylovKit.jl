module KrylovKitChainRulesCoreExt

using KrylovKit
using ChainRulesCore
using LinearAlgebra
using VectorInterface

include("utilities.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("svdsolve.jl")

end # module
