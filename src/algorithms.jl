# In the various algorithms we store the tolerance as a generic Real in order to
# construct these objects using keyword arguments in a type-stable manner. At
# the beginning of the corresponding algorithm, the actual tolerance will be
# converted to the concrete numeric type appropriate for the problem at hand


# Orthogonalization and orthonormalization
abstract Orthogonalizer
abstract Reorthogonalizer <: Orthogonalizer

# Simple
immutable ClassicalGramSchmidt <: Orthogonalizer
end

immutable ModifiedGramSchmidt <: Orthogonalizer
end

# A single reorthogonalization always
immutable ClassicalGramSchmidt2 <: Reorthogonalizer
end

immutable ModifiedGramSchmidt2 <: Reorthogonalizer
end

# Iterative reorthogonalization
immutable ClassicalGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end

immutable ModifiedGramSchmidtIR{S<:Real} <: Reorthogonalizer
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
abstract Factorizer

immutable LanczosIteration <: Factorizer
    krylovdim::Int
    tol::Real
end
immutable ArnoldiIteration <: Factorizer
    krylovdim::Int
    tol::Real
end

# Solving linear systems or eigenvalue problems
immutable SimpleLanczos{O}
    krylovdim::Int
    tol::Real
    orth::O
end
SimpleLanczos(orth::Orthogonalizer = orthdefault; krylovdim=30, tol=1e-12) = SimpleLanczos(krylovdim, tol, orth)

immutable RestartedLanczos{O}
    krylovdim::Int
    maxiter::Int
    tol::Real
    orth::O
end
RestartedLanczos(orth::Orthogonalizer = orthdefault; krylovdim=30, maxiter=100, tol=1e-12) =
    RestartedLanczos(krylovdim, maxiter, tol, orth)

immutable SimpleArnoldi{O}
    krylovdim::Int
    tol::Real
    orth::O
end
SimpleArnoldi(orth::Orthogonalizer = orthdefault; krylovdim=30, tol=1e-12) = SimpleArnoldi(krylovdim, tol, orth)

immutable RestartedArnoldi{O}
    krylovdim::Int
    maxiter::Int
    tol::Real
    orth::O
end
RestartedArnoldi(orth::Orthogonalizer = orthdefault; krylovdim=30, maxiter=100, tol=1e-12) =
    RestartedArnoldi(krylovdim, maxiter, tol, orth)

# Solving linear systems specifically
immutable CG
    maxiter::Int
    tol::Real
    reltol::Real
end
immutable SYMMLQ
    maxiter::Int
    tol::Real
    reltol::Real
end
immutable MINRES
    maxiter::Int
    tol::Real
    reltol::Real
end

immutable GMRES{O<:Orthogonalizer}
    krylovdim::Int
    maxiter::Int
    tol::Real
    reltol::Real
    orth::O
end

GMRES(orth::Orthogonalizer = orthdefault; tol=1e-12, reltol=1e-12, krylovdim=30, maxiter=100) =
    GMRES(krylovdim, maxiter, tol, reltol, orth)
