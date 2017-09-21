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

const orthdefault = mgsr # conservative choice

# Partial factorization: preliminary step for some linear or eigenvalue solvers
abstract type Factorizer end

struct LanczosIteration <: Factorizer
    krylovdim::Int
    tol::Real
end
struct ArnoldiIteration <: Factorizer
    krylovdim::Int
    tol::Real
end

# Solving eigenvalue problems
abstract type EigenSolver end

abstract type RestartStrategy end
struct NoRestart <: RestartStrategy
end
struct ExplicitRestart <: RestartStrategy
    maxiter::Int
end
struct ImplicitRestart <: RestartStrategy
    maxiter::Int
end
const norestart = NoRestart()

struct Lanczos{R,O} <: EigenSolver
    restart::R
    orth::O
    krylovdim::Int
    tol::Real
end
Lanczos(restart::RestartStrategy = ExplicitRestart(100), orth::Orthogonalizer = orthdefault; krylovdim=30, tol=1e-12) =
    Lanczos(restart, orth, krylovdim, tol)

struct Arnoldi{R,O} <: EigenSolver
    restart::R
    orth::O
    krylovdim::Int
    tol::Real
end
Arnoldi(restart::RestartStrategy = ExplicitRestart(100), orth::Orthogonalizer = orthdefault; krylovdim=30, tol=1e-12) =
    Arnoldi(restart, orth, krylovdim, tol)

# Solving linear systems specifically
abstract type LinearSolver end

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

struct GMRES{O<:Orthogonalizer} <: LinearSolver
    krylovdim::Int
    maxiter::Int
    tol::Real
    reltol::Real
    orth::O
end

GMRES(orth::Orthogonalizer = orthdefault; tol=1e-12, reltol=1e-12, krylovdim=30, maxiter=100) =
    GMRES(krylovdim, maxiter, tol, reltol, orth)
