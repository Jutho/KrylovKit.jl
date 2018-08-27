# In the various algorithms we store the tolerance as a generic Real in order to
# construct these objects using keyword arguments in a type-stable manner. At
# the beginning of the corresponding algorithm, the actual tolerance will be
# converted to the concrete numeric type appropriate for the problem at hand


# Orthogonalization and orthonormalization
"""
    abstract type Orthogonalizer

Supertype of a hierarchy for representing different orthogonalization strategies
or algorithms.

See also: [`ClassicalGramSchmidt`](@ref), [`ModifiedGramSchmidt`](@ref), [`ClassicalGramSchmidt2`](@ref),
    [`ModifiedGramSchmidt2`](@ref), [`ClassicalGramSchmidtIR`](@ref), [`ModifiedGramSchmidtIR`](@ref).
"""
abstract type Orthogonalizer end
abstract type Reorthogonalizer <: Orthogonalizer end

# Simple
"""
    ClassicalGramSchmidt()

Represents the classical Gram Schmidt algorithm for orthogonalizing different
vectors, typically not an optimal choice.
"""
struct ClassicalGramSchmidt <: Orthogonalizer
end

"""
    ModifiedGramSchmidt()

Represents the modified Gram Schmidt algorithm for orthogonalizing different
vectors, typically a reasonable choice for linear systems but not for eigenvalue
solvers with a large Krylov dimension.
"""
struct ModifiedGramSchmidt <: Orthogonalizer
end

# A single reorthogonalization always
"""
    ClassicalGramSchmidt2()

Represents the classical Gram Schmidt algorithm with a second reorthogonalization
step always taking place.
"""
struct ClassicalGramSchmidt2 <: Reorthogonalizer
end

"""
    ModifiedGramSchmidt2()

Represents the modified Gram Schmidt algorithm with a second reorthogonalization
step always taking place.
"""
struct ModifiedGramSchmidt2 <: Reorthogonalizer
end

# Iterative reorthogonalization
"""
    ClassicalGramSchmidtIR(η::Real)

Represents the classical Gram Schmidt algorithm with zero or more reorthogonalization
steps being applied untill the norm of the vector after an orthogonalization step
has not decreased by a factor smaller than `η` with respect to the norm before the step.
"""
struct ClassicalGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end

"""
    ModifiedGramSchmidtIR(η::Real)

Represents the modified Gram Schmidt algorithm with zero or more reorthogonalization
steps being applied untill the norm of the vector after an orthogonalization step
has not decreased by a factor smaller than `η` with respect to the norm before the step.
"""
struct ModifiedGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end

# Default values
module KrylovDefaults
    using ..KrylovKit
    const orth = KrylovKit.ModifiedGramSchmidtIR(1/sqrt(2)) # conservative choice
    const krylovdim = 30
    const maxiter = 100
    const tol = 1e-12
end

# Solving eigenvalue problems
abstract type KrylovAlgorithm end

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
"""
    Lanczos(orth::Orthogonalizer = KrylovDefaults.orth; krylovdim = KrylovDefaults.krylovdim,
        maxiter::Int = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)

Represents the Lanczos algorithm for building the Krylov subspace; assumes the
linear operator is real symmetric or complex Hermitian. Can be used in `factorize`,
`eigsolve` and `exponentiate`. The corresponding algorithms will build a Krylov
subspace of size at most `krylovdim`, which will be repeated at most `maxiter` times
and will stop when the norm of the residual of the Lanczos factorization is smaller
than `tol`. The orthogonalizer `orth` will be used to orthogonalize the different
Krylov vectors.

Use `Arnoldi` for non-symmetric or non-Hermitian linear operators.

See also: `factorize`, `eigsolve`, `exponentiate`, `Arnoldi`, `Orthogonalizer`
"""
struct Lanczos{O<:Orthogonalizer} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::Real
end
Lanczos(orth::Orthogonalizer = KrylovDefaults.orth; krylovdim = KrylovDefaults.krylovdim,
    maxiter::Int = KrylovDefaults.maxiter, tol = KrylovDefaults.tol) =
    Lanczos(orth, krylovdim, maxiter, tol)

"""
    Arnoldi(orth::Orthogonalizer = KrylovDefaults.orth; krylovdim = KrylovDefaults.krylovdim,
        maxiter::Int = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)

Represents the Arnoldi algorithm for building the Krylov subspace for a general.
matrix or linear operator. Can be used in `factorize`, `eigsolve` and `exponentiate`.
The corresponding algorithms will build a Krylov subspace of size at most `krylovdim`,
which will be repeated at most `maxiter` times and will stop when the norm of the
residual of the Arnoldi factorization is smaller than `tol`. The orthogonalizer
`orth` will be used to orthogonalize the different Krylov vectors.

Use `Lanczos` for real symmetric or complex Hermitian linear operators.

See also: `factorize`, `eigsolve`, `exponentiate`, `Lanczos`, `Orthogonalizer`
"""
struct Arnoldi{O<:Orthogonalizer} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::Real
end
Arnoldi(orth::Orthogonalizer = KrylovDefaults.orth; krylovdim = KrylovDefaults.krylovdim,
    maxiter::Int = KrylovDefaults.maxiter, tol = KrylovDefaults.tol) =
    Arnoldi(orth, krylovdim, maxiter, tol)

# Solving linear systems specifically
abstract type LinearSolver <: KrylovAlgorithm end

struct CG <: LinearSolver
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

"""
    GMRES(orth::Orthogonalizer = KrylovDefaults.orth; tol = KrylovDefaults.tol, reltol = KrylovDefaults.tol,
        krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter)

    Construct an instance of the GMRES algorithm with specified parameters, which
    can be passed to `linsolve` in order to iteratively solve a linear system. The
    `GMRES` method will search for the optimal `x` in a Krylov subspace of maximal
    size `krylovdim`, and repeat this process for at most `maxiter` times, or stop
    when ``| A*x - b | < max(tol, reltol*|b|)``.

    In building the Krylov subspace, it will use the orthogonalizer `orth`.

    Note that we do not follow the nomenclature in the traditional literature on `GMRES`,
    where `krylovdim` is referred to as the restart parameter, and every new Krylov
    vector counts as an iteration. I.e. our iteration count should rougly be multiplied
    by `krylovdim` to obtain the conventional iteration count.

    See also: [`linsolve`](@ref), [`CG`](@ref), [`MINRES`](@ref), [`BiCG`](@ref), [`BiCGStab`](@ref)
"""
struct GMRES{O<:Orthogonalizer} <: LinearSolver
    orth::O
    maxiter::Int
    krylovdim::Int
    tol::Real
    reltol::Real
end

GMRES(orth::Orthogonalizer = KrylovDefaults.orth; tol = KrylovDefaults.tol, reltol = KrylovDefaults.tol,
    krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter) =
    GMRES(orth, maxiter, krylovdim, tol, reltol)

# Solving eigenvalue systems specifically
abstract type EigenSolver <: KrylovAlgorithm end

struct JacobiDavidson <: EigenSolver
end
