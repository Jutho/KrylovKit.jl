using Random
using Test, TestExtras
using LinearAlgebra
using KrylovKit

include("testsetup.jl")
Random.seed!(76543210)

# Different vector wrappings
# --------------------------

module PureVecs
using Random
using Test, TestExtras
using LinearAlgebra
using KrylovKit
using ..TestSetup: precision, ≊, MinimalVec

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

wrapvec(v) = v
unwrapvec(v) = v
wrapvec2(v) = v
unwrapvec2(v) = v
wrapop(A::AbstractMatrix) = A

t = time()
include("factorize.jl")
include("gklfactorize.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("schursolve.jl")
include("geneigsolve.jl")
include("svdsolve.jl")
include("expintegrator.jl")
t = time() - t
println("Vector type $name: tests finished in $t seconds")

end

module InplaceVecs
using Random
using Test, TestExtras
using LinearAlgebra
using KrylovKit
using ..TestSetup: precision, ≊, MinimalVec

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

const inplace = true

wrapvec(v) = MinimalVec(v; inplace)
unwrapvec(v::MinimalVec) = getindex(v)
wrapvec2(v) = MinimalVec(v; inplace)
unwrapvec2(v::MinimalVec) = getindex(v)
wrapop(A::AbstractMatrix) = function (v, flag=Val(false))
    if flag === Val(true)
        return wrapvec(A' * unwrapvec2(v))
    else
        return wrapvec2(A * unwrapvec(v))
    end
end

t = time()
include("factorize.jl")
include("gklfactorize.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("schursolve.jl")
include("geneigsolve.jl")
include("svdsolve.jl")
include("expintegrator.jl")
t = time() - t
println("Vector type $name: tests finished in $t seconds")

end

module OutplaceVecs
using Random
using Test, TestExtras
using LinearAlgebra
using KrylovKit
using ..TestSetup: precision, ≊, MinimalVec

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

const inplace = false

wrapvec(v) = MinimalVec(v; inplace)
unwrapvec(v::MinimalVec) = getindex(v)
wrapvec2(v) = MinimalVec(v; inplace)
unwrapvec2(v::MinimalVec) = getindex(v)
wrapop(A::AbstractMatrix) = function (v, flag=Val(false))
    if flag === Val(true)
        return wrapvec(A' * unwrapvec2(v))
    else
        return wrapvec2(A * unwrapvec(v))
    end
end

t = time()
include("factorize.jl")
include("gklfactorize.jl")
include("linsolve.jl")
include("eigsolve.jl")
include("schursolve.jl")
include("geneigsolve.jl")
include("svdsolve.jl")
include("expintegrator.jl")
t = time() - t
println("Vector type $name: tests finished in $t seconds")

end

module MixedSVD

using Test, TestExtras
using LinearAlgebra
using Random
using KrylovKit
using ..TestSetup: precision, ≊, MinimalVec

wrapvec(v) = MinimalVec(v)
unwrapvec(v::MinimalVec) = v.vec
wrapvec2(v) = reshape(v, (length(v), 1)) # vector type 2 is a n x 1 Matrix
unwrapvec2(v) = reshape(v, (length(v),))
function wrapop(A::AbstractMatrix)
    return (x -> wrapvec2(A * unwrapvec(x)), y -> wrapvec(A' * unwrapvec2(y)))
end

const n = 10
const N = 100

const η₀ = 0.75 # seems to be necessary to get sufficient convergence for GKL iteration with Float32 precision
const cgs = ClassicalGramSchmidt()
const mgs = ModifiedGramSchmidt()
const cgs2 = ClassicalGramSchmidt2()
const mgs2 = ModifiedGramSchmidt2()
const cgsr = ClassicalGramSchmidtIR(η₀)
const mgsr = ModifiedGramSchmidtIR(η₀)

t = time()
include("gklfactorize.jl")
include("svdsolve.jl")
t = time() - t
println("Mixed vector type for GKL/SVD: tests finisthed in $t seconds")

end

module ExtrasTest
using Test, TestExtras
using LinearAlgebra
using Random
using KrylovKit
using ..TestSetup: precision, ≊, MinimalVec

const n = 10
const N = 100

const η₀ = 0.75 # seems to be necessary to get sufficient convergence for GKL iteration with Float32 precision
const cgs = ClassicalGramSchmidt()
const mgs = ModifiedGramSchmidt()
const cgs2 = ClassicalGramSchmidt2()
const mgs2 = ModifiedGramSchmidt2()
const cgsr = ClassicalGramSchmidtIR(η₀)
const mgsr = ModifiedGramSchmidtIR(η₀)

include("linalg.jl")
include("recursivevec.jl")
end

if VERSION >= v"1.6"
    include("ad.jl")
end

module AquaTests
using KrylovKit
using Aqua
Aqua.test_all(KrylovKit; ambiguities=false)
# treat ambiguities special because of ambiguities between ChainRulesCore and Base
if VERSION >= v"1.6" # ChainRulesCore leads to more ambiguities on julia < v1.6
    Aqua.test_ambiguities([KrylovKit, Base, Core]; exclude=[Base.:(==)])
end
end
