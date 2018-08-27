using Test
using LinearAlgebra
using Random
using KrylovKit

const n = 10
const N = 100

const η₀   = 1/sqrt(2) # conservative choice, probably 1/2 is sufficient
const cgs = ClassicalGramSchmidt()
const mgs = ModifiedGramSchmidt()
const cgs2 = ClassicalGramSchmidt2()
const mgs2 = ModifiedGramSchmidt2()
const cgsr = ClassicalGramSchmidtIR(η₀)
const mgsr = ModifiedGramSchmidtIR(η₀)

Random.seed!(1234567);

include("linalg.jl")
include("factorize.jl")
include("eigsolve.jl")
include("gmres.jl")
include("exponentiate.jl")
include("recursivevec.jl")
