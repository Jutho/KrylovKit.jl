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

Random.seed!(12345)

include("linalg.jl")

module PureVecs
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

    wrapvec(v) = v
    unwrapvec(v) = v
    wrapop(A::AbstractMatrix) = A

    t = time()
    include("factorize.jl")
    include("linsolve.jl")
    include("eigsolve.jl")
    include("schursolve.jl")
    include("svdsolve.jl")
    include("expintegrator.jl")
    t = time() - t
    println("Julia Vector type: tests finisthed in $t seconds")
end

module MinimalVecs
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

    include("minimalvec.jl")

    wrapvec(v) = MinimalVec(v)
    unwrapvec(v::MinimalVec) = getindex(v)
    wrapop(A::AbstractMatrix) =
        (v, flag=false)->wrapvec(flag ? A'*unwrapvec(v) : A*unwrapvec(v))

    t = time()
    include("factorize.jl")
    include("linsolve.jl")
    include("eigsolve.jl")
    include("schursolve.jl")
    include("svdsolve.jl")
    include("expintegrator.jl")
    t = time() - t
    println("Minimal vector type: tests finisthed in $t seconds")
end

include("recursivevec.jl")
