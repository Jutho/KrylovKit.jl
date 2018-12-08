# In the various algorithms we store the tolerance as a generic Real in order to
# construct these objects using keyword arguments in a type-stable manner. At
# the beginning of the corresponding algorithm, the actual tolerance will be
# converted to the concrete numeric type appropriate for the problem at hand


# Orthogonalization and orthonormalization
"""
    abstract type Orthogonalizer

Supertype of a hierarchy for representing different orthogonalization strategies or
algorithms.

See also: [`ClassicalGramSchmidt`](@ref), [`ModifiedGramSchmidt`](@ref),
[`ClassicalGramSchmidt2`](@ref), [`ModifiedGramSchmidt2`](@ref),
[`ClassicalGramSchmidtIR`](@ref), [`ModifiedGramSchmidtIR`](@ref).
"""
abstract type Orthogonalizer end
abstract type Reorthogonalizer <: Orthogonalizer end

# Simple
"""
    ClassicalGramSchmidt()

Represents the classical Gram Schmidt algorithm for orthogonalizing different vectors,
typically not an optimal choice.
"""
struct ClassicalGramSchmidt <: Orthogonalizer
end

"""
    ModifiedGramSchmidt()

Represents the modified Gram Schmidt algorithm for orthogonalizing different vectors,
typically a reasonable choice for linear systems but not for eigenvalue solvers with a
large Krylov dimension.
"""
struct ModifiedGramSchmidt <: Orthogonalizer
end

# A single reorthogonalization always
"""
    ClassicalGramSchmidt2()

Represents the classical Gram Schmidt algorithm with a second reorthogonalization step
always taking place.
"""
struct ClassicalGramSchmidt2 <: Reorthogonalizer
end

"""
    ModifiedGramSchmidt2()

Represents the modified Gram Schmidt algorithm with a second reorthogonalization step
always taking place.
"""
struct ModifiedGramSchmidt2 <: Reorthogonalizer
end

# Iterative reorthogonalization
"""
    ClassicalGramSchmidtIR(η::Real = 1/sqrt(2))

Represents the classical Gram Schmidt algorithm with iterative (i.e. zero or more)
reorthogonalization until the norm of the vector after an orthogonalization step has not
decreased by a factor smaller than `η` with respect to the norm before the step. The
default value corresponds to the Daniel-Gragg-Kaufman-Stewart condition.
"""
struct ClassicalGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end
ClassicalGramSchmidtIR() = ClassicalGramSchmidtIR(1/sqrt(2)) # Daniel-Gragg-Kaufman-Stewart

"""
    ModifiedGramSchmidtIR(η::Real = 1/sqrt(2))

Represents the modified Gram Schmidt algorithm with iterative (i.e. zero or more)
reorthogonalization until the norm of the vector after an orthogonalization step has not
decreased by a factor smaller than `η` with respect to the norm before the step. The
default value corresponds to the Daniel-Gragg-Kaufman-Stewart condition.
"""
struct ModifiedGramSchmidtIR{S<:Real} <: Reorthogonalizer
    η::S
end
ModifiedGramSchmidtIR() = ModifiedGramSchmidtIR(1/sqrt(2)) # Daniel-Gragg-Kaufman-Stewart

# Solving eigenvalue problems
abstract type KrylovAlgorithm end

# General purpose; good for linear systems, eigensystems and matrix functions
"""
    Lanczos(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,
        tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)

Represents the Lanczos algorithm for building the Krylov subspace; assumes the linear
operator is real symmetric or complex Hermitian. Can be used in `eigsolve` and
`exponentiate`. The corresponding algorithms will build a Krylov subspace of size at most
`krylovdim`, which will be repeated at most `maxiter` times and will stop when the norm of
the residual of the Lanczos factorization is smaller than `tol`. The orthogonalizer `orth`
will be used to orthogonalize the different Krylov vectors. Default verbosity level `verbosity`
is zero, meaning that no output will be printed.

Use `Arnoldi` for non-symmetric or non-Hermitian linear operators.

See also: `factorize`, `eigsolve`, `exponentiate`, `Arnoldi`, `Orthogonalizer`
"""
struct Lanczos{O<:Orthogonalizer, S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    verbosity::Int
end
Lanczos(; krylovdim::Int = KrylovDefaults.krylovdim,
            maxiter::Int = KrylovDefaults.maxiter,
            tol::Real = KrylovDefaults.tol,
            orth::Orthogonalizer = KrylovDefaults.orth,
            verbosity::Int = 0) =
    Lanczos(orth, krylovdim, maxiter, tol, verbosity)

"""
    GKL(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,
        tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)

Represents the Golub-Kahan-Lanczos bidiagonalization algorithm for sequentially building a
Krylov-like factorization of a genereal matrix or linear operator with a bidiagonal reduced
matrix. Can be used in `svdsolve`. The corresponding algorithm builds a Krylov subspace of
size at most `krylovdim`, which will be repeated at most `maxiter` times and will stop when
the norm of the residual of the Arnoldi factorization is smaller than `tol`. The
orthogonalizer `orth` will be used to orthogonalize the different Krylov vectors. Default
verbosity level `verbosity` is zero, meaning that no output will be printed.

See also: `svdsolve`, `Orthogonalizer`
"""
struct GKL{O<:Orthogonalizer, S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    verbosity::Int
end
GKL(; krylovdim::Int = KrylovDefaults.krylovdim,
        maxiter::Int = KrylovDefaults.maxiter,
        tol::Real = KrylovDefaults.tol,
        orth::Orthogonalizer = KrylovDefaults.orth,
        verbosity::Int = 0) =
    GKL(orth, krylovdim, maxiter, tol, verbosity)

"""
    Arnoldi(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,
        tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)

Represents the Arnoldi algorithm for building the Krylov subspace for a general matrix or
linear operator. Can be used in `eigsolve` and `exponentiate`. The corresponding algorithms
will build a Krylov subspace of size at most `krylovdim`, which will be repeated at most
`maxiter` times and will stop when the norm of the residual of the Arnoldi factorization is
smaller than `tol`. The orthogonalizer `orth` will be used to orthogonalize the different
Krylov vectors. Default verbosity level `verbosity` is zero, meaning that no output will be
printed.

Use `Lanczos` for real symmetric or complex Hermitian linear operators.

See also: [`eigsolve`](@ref), [`exponentiate`](@ref), [`Lanczos`](@ref),
[`Orthogonalizer`](@ref)
"""
struct Arnoldi{O<:Orthogonalizer, S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    verbosity::Int
end
Arnoldi(; krylovdim::Int = KrylovDefaults.krylovdim,
            maxiter::Int = KrylovDefaults.maxiter,
            tol::Real = KrylovDefaults.tol,
            orth::Orthogonalizer = KrylovDefaults.orth,
            verbosity::Int = 0) =
    Arnoldi(orth, krylovdim, maxiter, tol, verbosity)

"""
    GolubYe(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,
        tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)

Represents the Golub-Ye algorithm for solving hermitian (symmetric) generalized eigenvalue
problems `A x = λ B x` with positive definite `B`, without the need for inverting `B`.
Builds a Krylov subspace of size `krylovdim` starting from an estimate `x` by acting with
`(A - ρ(x) B)`, where `ρ(x) = dot(x, A*x)/dot(x, B*x)`, and employing the Lanczos
algorithm. This process is repeated at most `maxiter` times. In every iteration `k>1`, the
subspace will also be expanded to size `krylovdim+1` by adding ``x_k - x_{k-1}``, which is
known as the LOPCG correction and was suggested by Money and Ye. With `krylovdim = 2`, this
algorithm becomes equivalent to `LOPCG`.
"""
struct GolubYe{O<:Orthogonalizer, S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    verbosity::Int
end
GolubYe(; krylovdim::Int = KrylovDefaults.krylovdim,
            maxiter::Int = KrylovDefaults.maxiter,
            tol::Real = KrylovDefaults.tol,
            orth::Orthogonalizer = KrylovDefaults.orth,
            verbosity::Int = 0) =
    GolubYe(orth, krylovdim, maxiter, tol, verbosity)

# Solving linear systems specifically
abstract type LinearSolver <: KrylovAlgorithm end

"""
    CG(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)

Construct an instance of the conjugate gradient algorithm with specified parameters, which
can be passed to `linsolve` in order to iteratively solve a linear system with a positive
definite (and thus symmetric or hermitian) coefficent matrix or operator. The `CG` method
will search for the optimal `x` in a Krylov subspace of maximal size `maxiter`, or stop when
`norm(A*x - b) < tol`. Default verbosity level `verbosity` is zero, meaning that no output
will be printed.

See also: [`linsolve`](@ref), [`MINRES`](@ref), [`GMRES`](@ref), [`BiCG`](@ref),
[`BiCGStab`](@ref)
"""
struct CG{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
CG(; maxiter::Integer = KrylovDefaults.maxiter,
        tol::Real = KrylovDefaults.tol,
        verbosity::Int = 0) =
    CG(maxiter, tol, verbosity)

"""
    GMRES(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,
        tol = KrylovDefaults.tol, orth::Orthogonalizer = KrylovDefaults.orth)

Construct an instance of the GMRES algorithm with specified parameters, which can be passed
to `linsolve` in order to iteratively solve a linear system. The `GMRES` method will search
for the optimal `x` in a Krylov subspace of maximal size `krylovdim`, and repeat this
process for at most `maxiter` times, or stop when `norm(A*x - b) < tol`. In building the
Krylov subspace, `GMRES` will use the orthogonalizer `orth`. Default verbosity level
`verbosity` is zero, meaning that no output will be printed.

Note that in the traditional nomenclature of `GMRES`, the parameter `krylovdim` is referred
to as the restart parameter, and `maxiter` is the number of outer iterations, i.e. restart
cycles. The total iteration count, i.e. the number of expansion steps, is roughly
`krylovdim` times the number of iterations.

See also: [`linsolve`](@ref), [`BiCG`](@ref), [`BiCGStab`](@ref), [`CG`](@ref),
[`MINRES`](@ref)
"""
struct GMRES{O<:Orthogonalizer,S<:Real} <: LinearSolver
    orth::O
    maxiter::Int
    krylovdim::Int
    tol::S
    verbosity::Int
end
GMRES(; krylovdim::Integer = KrylovDefaults.krylovdim,
        maxiter::Integer = KrylovDefaults.maxiter,
        tol::Real = KrylovDefaults.tol,
        orth::Orthogonalizer = KrylovDefaults.orth,
        verbosity::Int = 0) =
    GMRES(orth, maxiter, krylovdim, tol, verbosity)

# TODO
"""
    MINRES(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)

Construct an instance of the conjugate gradient algorithm with specified parameters, which
can be passed to `linsolve` in order to iteratively solve a linear system with a real
symmetric or complex hermitian coefficent matrix or operator. The `MINRES` method will
search for the optimal `x` in a Krylov subspace of maximal size `maxiter`, or stop when
`norm(A*x - b) < tol`. In building the Krylov subspace, `MINRES` will use the
orthogonalizer `orth`. Default verbosity level `verbosity` is zero, meaning that no output
will be printed.

!!! warning "Not implemented yet"

See also: [`linsolve`](@ref), [`CG`](@ref), [`GMRES`](@ref), [`BiCG`](@ref), [`BiCGStab`](@ref)
"""
struct MINRES{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
MINRES(; maxiter::Integer = KrylovDefaults.maxiter,
            tol::Real = KrylovDefaults.tol,
            verbosity::Int = 0) =
    MINRES(maxiter, tol, verbosity)

"""
    BiCG(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)

Construct an instance of the Biconjugate gradient algorithm with specified parameters, which
can be passed to `linsolve` in order to iteratively solve a linear system general linear
map, of which the adjoint can also be applied. The `BiCG` method will search for the
optimal `x` in a Krylov subspace of maximal size `maxiter`, or stop when `norm(A*x - b) <
tol`. Default verbosity level `verbosity` is zero, meaning that no output will be printed.

!!! warning "Not implemented yet"

See also: [`linsolve`](@ref), [`BiCGStab`](@ref), [`GMRES`](@ref), [`CG`](@ref),
[`MINRES`](@ref)
"""
struct BiCG{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
BiCG(; maxiter::Integer = KrylovDefaults.maxiter,
        tol::Real = KrylovDefaults.tol,
        verbosity::Int = 0) =
    BiCG(maxiter, tol, verbosity)


"""
    BiCGStab(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)

Construct an instance of the Biconjugate gradient algorithm with specified parameters, which
can be passed to `linsolve` in order to iteratively solve a linear system general linear
map. The `BiCGStab` method will search for the optimal `x` in a Krylov subspace of maximal
size `maxiter`, or stop when `norm(A*x - b) < tol`. Default verbosity level `verbosity` is
zero, meaning that no output will be printed.

!!! warning "Not implemented yet"

See also: [`linsolve`](@ref), [`BiCG`](@ref), [`GMRES`](@ref), [`CG`](@ref),
[`MINRES`](@ref)
"""
struct BiCGStab{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
BiCGStab(; maxiter::Integer = KrylovDefaults.maxiter,
            tol::Real = KrylovDefaults.tol,
            verbosity::Int = 0) =
    BiCGStab(maxiter, tol, verbosity)


# Solving eigenvalue systems specifically
abstract type EigenSolver <: KrylovAlgorithm end

struct JacobiDavidson <: EigenSolver
end

# Default values
"""
    module KrylovDefaults
        const orth = KrylovKit.ModifiedGramSchmidtIR()
        const krylovdim = 30
        const maxiter = 100
        const tol = 1e-12
    end

A module listing the default values for the typical parameters in Krylov based algorithms:
*   `orth`: the orthogonalization routine used to orthogonalize the Krylov basis in the
    `Lanczos` or `Arnoldi` iteration
*   `krylovdim`: the maximal dimension of the Krylov subspace that will be constructed
*   `maxiter`: the maximal number of outer iterations, i.e. the maximum number of times the
    Krylov subspace may be rebuilt
*   `tol`: the tolerance to which the problem must be solved, based on a suitable error
    measure, e.g. the norm of some residual.
!!! warning
    The default value of `tol` is a `Float64` value, if you solve problems in `Float32` or
    `ComplexF32` arithmetic, you should always specify a new `tol` as the default value
    will not be attainable.
"""
module KrylovDefaults
    using ..KrylovKit
    const orth = KrylovKit.ModifiedGramSchmidt2() # conservative choice
    const krylovdim = 30
    const maxiter = 100
    const tol = 1e-12
end
