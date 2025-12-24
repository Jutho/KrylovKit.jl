using Random
Random.seed!(76543210)

using Test, TestExtras, Logging
using LinearAlgebra, SparseArrays
using KrylovKit
using KrylovKit: SILENT_LEVEL, WARN_LEVEL, STARTSTOP_LEVEL, EACHITERATION_LEVEL
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
@testset "Krylov factorisations" verbose = true begin
    include("factorize.jl")
end
@testset "Linear problems with linsolve" verbose = true begin
    include("linsolve.jl")
end
@testset "Least squares problems with lssolve" verbose = true begin
    include("lssolve.jl")
end
@testset "Eigenvalue problems with eigsolve" verbose = true begin
    include("eigsolve.jl")
    include("schursolve.jl")
    include("geneigsolve.jl")
end
@testset "Biorthogonal eigenvalue problems with bieigsolve" verbose = true begin
    include("bieigsolve.jl")
end
@testset "Singular value problems with svdsolve" verbose = true begin
    include("svdsolve.jl")
end
@testset "Exponentiate and exponential integrator" verbose = true begin
    include("expintegrator.jl")
end
@testset "Linear Algebra Utilities" verbose = true begin
    include("linalg.jl")
end
@testset "Singular value problems via eigsolve with nested tuples" verbose = true begin
    include("nestedtuple.jl")
end

@testset "Linsolve differentiation rules" verbose = true begin
    include("ad/linsolve.jl")
end
@testset "Eigsolve differentiation rules" verbose = true begin
    include("ad/eigsolve.jl")
    include("ad/repeatedeigsolve.jl")
end
@testset "Svdsolve differentiation rules" verbose = true begin
    include("ad/svdsolve.jl")
end
@testset "block" verbose = true begin
    include("block.jl")
end
t = time() - t

# Issues
# ------
@testset "Known issues" verbose = true begin
    include("issues.jl")
end
println("Tests finished in $t seconds")

module AquaTests
    using KrylovKit
    using Aqua
    Aqua.test_all(KrylovKit; ambiguities = false)
    # treat ambiguities special because of ambiguities between ChainRulesCore and Base
    Aqua.test_ambiguities([KrylovKit, Base, Core]; exclude = [Base.:(==)])

end
