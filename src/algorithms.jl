# In the various algorithms we store the tolerance as a generic Real in order to
# construct these objects using keyword arguments in a type-stable manner. At
# the beginning of the corresponding algorithm, the actual tolerance will be
# converted to the concrete numeric type appropriate for the problem at hand


# Orthogonalization and orthonormalization
abstract type Orthogonalizer end
abstract type Reorthogonalizer <: Orthogonalizer end

# Simple
struct ClassicalGramSchmidt <: Orthogonalizer
end

struct ModifiedGramSchmidt <: Orthogonalizer
end

# A single reorthogonalization always
struct ClassicalGramSchmidt2 <: Reorthogonalizer
end

struct ModifiedGramSchmidt2 <: Reorthogonalizer
end

# Iterative reorthogonalization
struct ClassicalGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end

struct ModifiedGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end

const η₀   = 1/sqrt(2) # conservative choice, probably 1/2 is sufficient

const cgs = ClassicalGramSchmidt()
const mgs = ModifiedGramSchmidt()
const cgs2 = ClassicalGramSchmidt2()
const mgs2 = ModifiedGramSchmidt2()
const cgsr = ClassicalGramSchmidtIR(η₀)
const mgsr = ModifiedGramSchmidtIR(η₀)

# Default values
module Defaults
    using ..KrylovKit
    const orth = KrylovKit.ModifiedGramSchmidtIR(1/sqrt(2)) # conservative choice
    const krylovdim = 30
    const maxiter = 100
    const tol = 1e-12
end

# Solving eigenvalue problems
abstract type Algorithm end

abstract type RestartStrategy end
struct NoRestart <: RestartStrategy
end
struct ExplicitRestart <: RestartStrategy
    maxiter::Int
end
struct ImplicitRestart <: RestartStrategy # only meaningful for eigenvalue problems
    maxiter::Int
end
const norestart = NoRestart()

# General purpose; good for linear systems, eigensystems and matrix functions
struct Lanczos{R<:RestartStrategy, O<:Orthogonalizer} <: Algorithm
    restart::R
    orth::O
    krylovdim::Int
    tol::Real
end
Lanczos(restart::RestartStrategy = ImplicitRestart(Defaults.maxiter), orth::Orthogonalizer = Defaults.orth; krylovdim = Defaults.krylovdim, tol = Defaults.tol) =
    Lanczos(restart, orth, krylovdim, tol)

struct Arnoldi{R<:RestartStrategy, O<:Orthogonalizer} <: Algorithm
    restart::R
    orth::O
    krylovdim::Int
    tol::Real
end
Arnoldi(restart::RestartStrategy = ImplicitRestart(Defaults.maxiter), orth::Orthogonalizer = Defaults.orth; krylovdim = Defaults.krylovdim, tol = Defaults.tol) =
    Arnoldi(restart, orth, krylovdim, tol)

# Solving linear systems specifically
abstract type LinearSolver <: Algorithm end

struct CG <: LinearSolver
    maxiter::Int
    tol::Real
    reltol::Real
end
struct SYMMLQ <: LinearSolver
    maxiter::Int
    tol::Real
    reltol::Real
end
struct MINRES <: LinearSolver
    maxiter::Int
    tol::Real
    reltol::Real
end
struct BiCG <: LinearSolver
    maxiter::Int
    tol::Real
    reltol::Real
end
struct BiCGStab <: LinearSolver
    maxiter::Int
    tol::Real
    reltol::Real
end

struct GMRES{O<:Orthogonalizer} <: LinearSolver
    orth::O
    maxiter::Int
    krylovdim::Int
    tol::Real
    reltol::Real
end

GMRES(orth::Orthogonalizer = Defaults.orth; tol = Defaults.tol, reltol = Defaults.tol, krylovdim = Defaults.krylovdim, maxiter = Defaults.maxiter) =
    GMRES(orth, maxiter, krylovdim, tol, reltol)

# Solving eigenvalue systems specifically
abstract type EigenSolver <: Algorithm end

struct JacobiDavidson <: EigenSolver
end
