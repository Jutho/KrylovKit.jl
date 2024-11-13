using Random
Random.seed!(76543210)

using Test, TestExtras
using LinearAlgebra
using KrylovKit
using VectorInterface

include("testsetup.jl")
using ..TestSetup

# Parameters
# ----------
const n = 10
const N = 100

const η₀ = 0.75 # seems to be necessary to get sufficient convergence for GKL iteration with Float32 precision
const cgs = ClassicalGramSchmidt()
const mgs = ModifiedGramSchmidt()
const cgs2 = ClassicalGramSchmidt2()
const mgs2 = ModifiedGramSchmidt2()
const cgsr = ClassicalGramSchmidtIR(η₀)
const mgsr = ModifiedGramSchmidtIR(η₀)

# Tests
# -----
t = time()
include("factorize.jl")
include("gklfactorize.jl")

include("linsolve.jl")
include("eigsolve.jl")
include("schursolve.jl")
include("geneigsolve.jl")
include("svdsolve.jl")
include("expintegrator.jl")

include("linalg.jl")
include("nestedtuple.jl")

include("ad/linsolve.jl")
include("ad/eigsolve.jl")
include("ad/degenerateeigsolve.jl")
include("ad/svdsolve.jl")

t = time() - t
println("Tests finished in $t seconds")

# Issues
# ------
include("issues.jl")

module AquaTests
using KrylovKit
using Aqua
Aqua.test_all(KrylovKit; ambiguities=false)
# treat ambiguities special because of ambiguities between ChainRulesCore and Base
Aqua.test_ambiguities([KrylovKit, Base, Core]; exclude=[Base.:(==)])

end
