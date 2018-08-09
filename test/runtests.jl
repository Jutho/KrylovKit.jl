using Test
using LinearAlgebra
using Random
using KrylovKit

const n = 10
const N = 100

include("linalg.jl")
include("factorize.jl")
include("eigsolve.jl")
include("gmres.jl")
include("exponentiate.jl")
