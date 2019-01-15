using Test
using LinearAlgebra
using Random
using KrylovKit

const n = 10
const N = 100

const η₀   = 0.75 # seems to be necessary to get sufficient convergence for GKL iteration with Float32 precision
const cgs = ClassicalGramSchmidt()
const mgs = ModifiedGramSchmidt()
const cgs2 = ClassicalGramSchmidt2()
const mgs2 = ModifiedGramSchmidt2()
const cgsr = ClassicalGramSchmidtIR(η₀)
const mgsr = ModifiedGramSchmidtIR(η₀)

Random.seed!(1234567)

include("linalg.jl")
include("factorize.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("schursolve.jl")
include("svdsolve.jl")
include("expintegrator.jl")
include("recursivevec.jl")
