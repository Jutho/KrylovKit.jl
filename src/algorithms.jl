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
struct ClassicalGramSchmidt <: Orthogonalizer end

"""
    ModifiedGramSchmidt()

Represents the modified Gram Schmidt algorithm for orthogonalizing different vectors,
typically a reasonable choice for linear systems but not for eigenvalue solvers with a
large Krylov dimension.
"""
struct ModifiedGramSchmidt <: Orthogonalizer end

# A single reorthogonalization always
"""
    ClassicalGramSchmidt2()

Represents the classical Gram Schmidt algorithm with a second reorthogonalization step
always taking place.
"""
struct ClassicalGramSchmidt2 <: Reorthogonalizer end

"""
    ModifiedGramSchmidt2()

Represents the modified Gram Schmidt algorithm with a second reorthogonalization step
always taking place.
"""
struct ModifiedGramSchmidt2 <: Reorthogonalizer end

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
ClassicalGramSchmidtIR() = ClassicalGramSchmidtIR(1 / sqrt(2)) # Daniel-Gragg-Kaufman-Stewart

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
ModifiedGramSchmidtIR() = ModifiedGramSchmidtIR(1 / sqrt(2)) # Daniel-Gragg-Kaufman-Stewart

# Solving eigenvalue problems
abstract type KrylovAlgorithm end

# General purpose; good for linear systems, eigensystems and matrix functions
"""
    Lanczos(; krylovdim=KrylovDefaults.krylovdim[],
            maxiter=KrylovDefaults.maxiter[],
            tol=KrylovDefaults.tol[],
            orth=KrylovDefaults.orth,
            eager=false,
            verbosity=KrylovDefaults.verbosity[])

Represents the Lanczos algorithm for building the Krylov subspace; assumes the linear
operator is real symmetric or complex Hermitian. Can be used in `eigsolve` and
`exponentiate`. The corresponding algorithms will build a Krylov subspace of size at most
`krylovdim`, which will be repeated at most `maxiter` times and will stop when the norm of
the residual of the Lanczos factorization is smaller than `tol`. The orthogonalizer `orth`
will be used to orthogonalize the different Krylov vectors. Eager mode, as selected by
`eager=true`, means that the algorithm that uses this Lanczos process (e.g. `eigsolve`)
can try to finish its computation before the total Krylov subspace of dimension `krylovdim`
is constructed. The default verbosity level `verbosity` amounts to printing warnings upon
lack of convergence.


Use `Arnoldi` for non-symmetric or non-Hermitian linear operators.

See also: [Factorization types](@ref), [`eigsolve`](@ref), [`exponentiate`](@ref), [`Arnoldi`](@ref), [`Orthogonalizer`](@ref)
"""
struct Lanczos{O<:Orthogonalizer,S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    eager::Bool
    verbosity::Int
end
function Lanczos(;
                 krylovdim::Int=KrylovDefaults.krylovdim[],
                 maxiter::Int=KrylovDefaults.maxiter[],
                 tol::Real=KrylovDefaults.tol[],
                 orth::Orthogonalizer=KrylovDefaults.orth,
                 eager::Bool=false,
                 verbosity::Int=KrylovDefaults.verbosity[])
    return Lanczos(orth, krylovdim, maxiter, tol, eager, verbosity)
end

"""
    BlockLanczos(; krylovdim=KrylovDefaults.blockkrylovdim[],
            maxiter=KrylovDefaults.maxiter[],
            tol=KrylovDefaults.tol[],
            orth=KrylovDefaults.orth,
            eager=false,
            verbosity=KrylovDefaults.verbosity[],
            qr_tol::Real=KrylovDefaults.tol[])

The block version of [`Lanczos`](@ref) is suited for solving eigenvalue problems with repeated extremal eigenvalues.
Its implementation is mainly based on *Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations* (4th ed., pp. 566–569).
Arguments `krylovdim`, `maxiter`, `tol`, `orth`, `eager` and `verbosity` are the same as `Lanczos`.
`qr_tol` is the error tolerance for `block_qr!` - a subroutine used to orthorgonalize the vectors in the same block.
The initial size of the block is determined by the number of vectors that a user provides in the starting block;
the size of the block can shrink during iterations.
The initial block size determines the maximum multiplicity of the target eigenvalue can that be resolved.

The iteration stops when either the norm of the residual is below `tol` or a sufficient number of eigenvectors have converged. [Reference](https://www.netlib.org/utk/people/JackDongarra/etemplates/node250.html)

Use `Arnoldi` for non-symmetric or non-Hermitian linear operators. 

See also: [Factorization types](@ref), [`eigsolve`](@ref), [`Arnoldi`](@ref), [`Orthogonalizer`](@ref)
"""
struct BlockLanczos{O<:Orthogonalizer,S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    qr_tol::S
    eager::Bool
    verbosity::Int
end
function BlockLanczos(;
                      krylovdim::Int=KrylovDefaults.blockkrylovdim[],
                      maxiter::Int=KrylovDefaults.maxiter[],
                      tol::Real=KrylovDefaults.tol[],
                      qr_tol::Real=KrylovDefaults.tol[],
                      orth::Orthogonalizer=KrylovDefaults.orth,
                      eager::Bool=false,
                      verbosity::Int=KrylovDefaults.verbosity[])
    return BlockLanczos(orth, krylovdim, maxiter, promote(tol, qr_tol)..., eager, verbosity)
end

"""
    GKL(; krylovdim=KrylovDefaults.krylovdim[],
        maxiter=KrylovDefaults.maxiter[],
        tol=KrylovDefaults.tol[],
        orth=KrylovDefaults.orth,
        eager=false,
        verbosity=KrylovDefaults.verbosity[])

Represents the Golub-Kahan-Lanczos bidiagonalization algorithm for sequentially building a
Krylov-like factorization of a general matrix or linear operator with a bidiagonal reduced
matrix. Can be used in `svdsolve`. The corresponding algorithm builds a Krylov subspace of
size at most `krylovdim`, which will be repeated at most `maxiter` times and will stop when
the norm of the residual of the Arnoldi factorization is smaller than `tol`. The
orthogonalizer `orth` will be used to orthogonalize the different Krylov vectors. The default
verbosity level `verbosity` amounts to printing warnings upon lack of convergence.


See also: [`svdsolve`](@ref), [`Orthogonalizer`](@ref)
"""
struct GKL{O<:Orthogonalizer,S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    eager::Bool
    verbosity::Int
end
function GKL(;
             krylovdim::Int=KrylovDefaults.krylovdim[],
             maxiter::Int=KrylovDefaults.maxiter[],
             tol::Real=KrylovDefaults.tol[],
             orth::Orthogonalizer=KrylovDefaults.orth,
             eager::Bool=false,
             verbosity::Int=KrylovDefaults.verbosity[])
    return GKL(orth, krylovdim, maxiter, tol, eager, verbosity)
end

"""
    Arnoldi(; krylovdim=KrylovDefaults.krylovdim[],
            maxiter=KrylovDefaults.maxiter[],
            tol=KrylovDefaults.tol[],
            orth=KrylovDefaults.orth,
            eager=false,
            verbosity=KrylovDefaults.verbosity[])

Represents the Arnoldi algorithm for building the Krylov subspace for a general matrix or
linear operator. Can be used in `eigsolve` and `exponentiate`. The corresponding algorithms
will build a Krylov subspace of size at most `krylovdim`, which will be repeated at most
`maxiter` times and will stop when the norm of the residual of the Arnoldi factorization is
smaller than `tol`. The orthogonalizer `orth` will be used to orthogonalize the different
Krylov vectors. Eager mode, as selected by `eager=true`, means that the algorithm that
uses this Arnoldi process (e.g. `eigsolve`) can try to finish its computation before the
total Krylov subspace of dimension `krylovdim` is constructed. The default verbosity level
`verbosity` amounts to printing warnings upon lack of convergence.


Use `Lanczos` for real symmetric or complex Hermitian linear operators.

See also: [`eigsolve`](@ref), [`exponentiate`](@ref), [`Lanczos`](@ref),
[`Orthogonalizer`](@ref)
"""
struct Arnoldi{O<:Orthogonalizer,S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    eager::Bool
    verbosity::Int
end
function Arnoldi(;
                 krylovdim::Int=KrylovDefaults.krylovdim[],
                 maxiter::Int=KrylovDefaults.maxiter[],
                 tol::Real=KrylovDefaults.tol[],
                 orth::Orthogonalizer=KrylovDefaults.orth,
                 eager::Bool=false,
                 verbosity::Int=KrylovDefaults.verbosity[])
    return Arnoldi(orth, krylovdim, maxiter, tol, eager, verbosity)
end

"""
    BiArnoldi(; krylovdim=KrylovDefaults.krylovdim[],
              maxiter=KrylovDefaults.maxiter[],
              tol=KrylovDefaults.tol[],
              orth=KrylovDefaults.orth,
              eager=false,
              verbosity=KrylovDefaults.verbosity[])

Represents the BiArnoldi algorithm for building the Krylov subspace for a general matrix or
linear operator. Can be used in `bieigsolve`. The corresponding algorithm will build a Krylov subspace of size at most `krylovdim`, which will be repeated at most `maxiter` times and
will stop when the norm of the residual of the Arnoldi factorization is smaller than `tol`.
The orthogonalizer `orth` will be used to orthogonalize the different Krylov vectors.
Eager mode, as selected by `eager=true`, means that the algorithm that uses this BiArnoldi
process (i.e. `bieigsolve`) can try to finish its computation before the total Krylov
subspace of dimension `krylovdim` is constructed. The default verbosity level
`verbosity` amounts to printing warnings upon lack of convergence.


See also: [`bieigsolve`](@ref), [`Orthogonalizer`](@ref)
"""
struct BiArnoldi{O<:Orthogonalizer,S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    eager::Bool
    verbosity::Int
end
function BiArnoldi(;
                   krylovdim::Int=KrylovDefaults.krylovdim[],
                   maxiter::Int=KrylovDefaults.maxiter[],
                   tol::Real=KrylovDefaults.tol[],
                   orth::Orthogonalizer=KrylovDefaults.orth,
                   eager::Bool=false,
                   verbosity::Int=KrylovDefaults.verbosity[])
    return BiArnoldi(orth, krylovdim, maxiter, tol, eager, verbosity)
end

"""
    GolubYe(; krylovdim=KrylovDefaults.krylovdim[],
            maxiter=KrylovDefaults.maxiter[],
            tol=KrylovDefaults.tol[],
            orth=KrylovDefaults.orth,
            eager=false,
            verbosity=KrylovDefaults.verbosity[])

Represents the Golub-Ye algorithm for solving hermitian (symmetric) generalized eigenvalue
problems `A x = λ B x` with positive definite `B`, without the need for inverting `B`.
Builds a Krylov subspace of size `krylovdim` starting from an estimate `x` by acting with
`(A - ρ(x) B)`, where `ρ(x) = dot(x, A*x)/dot(x, B*x)`, and employing the Lanczos
algorithm. This process is repeated at most `maxiter` times. In every iteration `k>1`, the
subspace will also be expanded to size `krylovdim+1` by adding ``x_k - x_{k-1}``, which is
known as the LOPCG correction and was suggested by Money and Ye. With `krylovdim=2`, this
algorithm becomes equivalent to `LOPCG`.
"""
struct GolubYe{O<:Orthogonalizer,S<:Real} <: KrylovAlgorithm
    orth::O
    krylovdim::Int
    maxiter::Int
    tol::S
    verbosity::Int
end
function GolubYe(;
                 krylovdim::Int=KrylovDefaults.krylovdim[],
                 maxiter::Int=KrylovDefaults.maxiter[],
                 tol::Real=KrylovDefaults.tol[],
                 orth::Orthogonalizer=KrylovDefaults.orth,
                 verbosity::Int=KrylovDefaults.verbosity[])
    return GolubYe(orth, krylovdim, maxiter, tol, verbosity)
end

# Solving linear systems specifically
abstract type LinearSolver <: KrylovAlgorithm end

"""
    CG(; maxiter=KrylovDefaults.maxiter[], tol=KrylovDefaults.tol[], verbosity=KrylovDefaults.verbosity[])

Construct an instance of the conjugate gradient algorithm with specified parameters, which
can be passed to `linsolve` in order to iteratively solve a linear system with a positive
definite (and thus symmetric or hermitian) coefficient matrix or operator. The `CG` method
will search for the optimal `x` in a Krylov subspace of maximal size `maxiter`, or stop when
`norm(A*x - b) < tol`. The default verbosity level `verbosity` amounts to printing warnings
upon lack of convergence.


See also: [`linsolve`](@ref), [`MINRES`](@ref), [`GMRES`](@ref), [`BiCG`](@ref), [`LSMR`](@ref),
[`BiCGStab`](@ref)
"""
struct CG{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
function CG(;
            maxiter::Integer=KrylovDefaults.maxiter[],
            tol::Real=KrylovDefaults.tol[],
            verbosity::Int=KrylovDefaults.verbosity[])
    return CG(maxiter, tol, verbosity)
end

"""
    GMRES(; krylovdim=KrylovDefaults.krylovdim[],
            maxiter=KrylovDefaults.maxiter[],
            tol=KrylovDefaults.tol[], 
            orth::Orthogonalizer=KrylovDefaults.orth,
            verbosity=KrylovDefaults.verbosity[])

Construct an instance of the GMRES algorithm with specified parameters, which can be passed
to `linsolve` in order to iteratively solve a linear system. The `GMRES` method will search
for the optimal `x` in a Krylov subspace of maximal size `krylovdim`, and repeat this
process for at most `maxiter` times, or stop when `norm(A*x - b) < tol`. In building the
Krylov subspace, `GMRES` will use the orthogonalizer `orth`. The default verbosity level
`verbosity` amounts to printing warnings upon lack of convergence.


Note that in the traditional nomenclature of `GMRES`, the parameter `krylovdim` is referred
to as the restart parameter, and `maxiter` is the number of outer iterations, i.e. restart
cycles. The total iteration count, i.e. the number of expansion steps, is roughly
`krylovdim` times the number of iterations.

See also: [`linsolve`](@ref), [`BiCG`](@ref), [`BiCGStab`](@ref), [`CG`](@ref), [`LSMR`](@ref),
[`MINRES`](@ref)
"""
struct GMRES{O<:Orthogonalizer,S<:Real} <: LinearSolver
    orth::O
    maxiter::Int
    krylovdim::Int
    tol::S
    verbosity::Int
end
function GMRES(;
               krylovdim::Integer=KrylovDefaults.krylovdim[],
               maxiter::Integer=KrylovDefaults.maxiter[],
               tol::Real=KrylovDefaults.tol[],
               orth::Orthogonalizer=KrylovDefaults.orth,
               verbosity::Int=KrylovDefaults.verbosity[])
    return GMRES(orth, maxiter, krylovdim, tol, verbosity)
end

# TODO
"""
    MINRES(; maxiter=KrylovDefaults.maxiter[], tol=KrylovDefaults.tol[], verbosity=KrylovDefaults.verbosity[])

    !!! warning "Not implemented yet"

    Construct an instance of the conjugate gradient algorithm with specified parameters,
    which can be passed to `linsolve` in order to iteratively solve a linear system with a
    real symmetric or complex hermitian coefficient matrix or operator. The `MINRES` method
    will search for the optimal `x` in a Krylov subspace of maximal size `maxiter`, or stop
    when `norm(A*x - b) < tol`. In building the Krylov subspace, `MINRES` will use the
    orthogonalizer `orth`. The default verbosity level `verbosity` amounts to printing
    warnings upon lack of convergence.


See also: [`linsolve`](@ref), [`CG`](@ref), [`GMRES`](@ref), [`BiCG`](@ref), [`LSMR`](@ref),
[`BiCGStab`](@ref)
"""
struct MINRES{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
function MINRES(;
                maxiter::Integer=KrylovDefaults.maxiter[],
                tol::Real=KrylovDefaults.tol[],
                verbosity::Int=KrylovDefaults.verbosity[])
    return MINRES(maxiter, tol, verbosity)
end

"""
    BiCG(; maxiter=KrylovDefaults.maxiter[], tol=KrylovDefaults.tol[], verbosity=KrylovDefaults.verbosity[])

    !!! warning "Not implemented yet"

    Construct an instance of the Biconjugate gradient algorithm with specified parameters,
    which can be passed to `linsolve` in order to iteratively solve a linear system general
    linear map, of which the adjoint can also be applied. The `BiCG` method will search for
    the optimal `x` in a Krylov subspace of maximal size `maxiter`, or stop when `norm(A*x -
    b) < tol`. The default verbosity level `verbosity` amounts to printing warnings upon
    lack of convergence.


See also: [`linsolve`](@ref), [`GMRES`](@ref), [`CG`](@ref), [`BiCGStab`](@ref), [`LSMR`](@ref),
[`MINRES`](@ref)
"""
struct BiCG{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
function BiCG(;
              maxiter::Integer=KrylovDefaults.maxiter[],
              tol::Real=KrylovDefaults.tol[],
              verbosity::Int=KrylovDefaults.verbosity[])
    return BiCG(maxiter, tol, verbosity)
end

"""
    BiCGStab(; maxiter=KrylovDefaults.maxiter[], tol=KrylovDefaults.tol[], verbosity=KrylovDefaults.verbosity[])

    Construct an instance of the Biconjugate gradient algorithm with specified parameters,
    which can be passed to `linsolve` in order to iteratively solve a linear system general
    linear map. The `BiCGStab` method will search for the optimal `x` in a Krylov subspace
    of maximal size `maxiter`, or stop when `norm(A*x - b) < tol`. The default verbosity level 
    `verbosity` amounts to printing warnings upon lack of convergence.

See also: [`linsolve`](@ref), [`GMRES`](@ref), [`CG`](@ref), [`BiCG`](@ref), [`LSMR`](@ref),
[`MINRES`](@ref)
"""
struct BiCGStab{S<:Real} <: LinearSolver
    maxiter::Int
    tol::S
    verbosity::Int
end
function BiCGStab(;
                  maxiter::Integer=KrylovDefaults.maxiter[],
                  tol::Real=KrylovDefaults.tol[],
                  verbosity::Int=KrylovDefaults.verbosity[])
    return BiCGStab(maxiter, tol, verbosity)
end

# Solving least squares problems
abstract type LeastSquaresSolver <: KrylovAlgorithm end
"""
    LSMR(; krylovdim=1,
            maxiter=KrylovDefaults.maxiter[],
            tol=KrylovDefaults.tol[], 
            orth::Orthogonalizer=ModifiedGramSchmidt(),
            verbosity=KrylovDefaults.verbosity[])

Represents the LSMR algorithm, which minimizes ``\\|Ax - b\\|^2 + \\|λx\\|^2`` in the Euclidean norm.
If multiple solutions exists the minimum norm solution is returned.
The method is based on the Golub-Kahan bidiagonalization process. It is
algebraically equivalent to applying MINRES to the normal equations
``(A^*A + λ^2I)x = A^*b``, but has better numerical properties,
especially if ``A`` is ill-conditioned.

The `LSMR` method will search for the optimal ``x`` in a Krylov subspace of maximal size 
`maxiter`, or stop when ``norm(A'*(A*x - b) + λ^2 * x) < tol``. The parameter `krylovdim`
does in this case not indicate that a subspace of that size will be built, but represents the
number of most recent vectors that will be kept to which the next vector will be reorthogonalized.
The default verbosity level `verbosity` amounts to printing warnings upon lack of convergence.

See also: [`lssolve`](@ref)
"""
struct LSMR{O<:Orthogonalizer,S<:Real} <: LeastSquaresSolver
    orth::O
    maxiter::Int
    krylovdim::Int
    tol::S
    verbosity::Int
end
function LSMR(;
              krylovdim::Integer=KrylovDefaults.krylovdim[],
              maxiter::Integer=KrylovDefaults.maxiter[],
              tol::Real=KrylovDefaults.tol[],
              orth::Orthogonalizer=ModifiedGramSchmidt(),
              verbosity::Int=KrylovDefaults.verbosity[])
    return LSMR(orth, maxiter, krylovdim, tol, verbosity)
end

# Solving eigenvalue systems specifically
abstract type EigenSolver <: KrylovAlgorithm end

struct JacobiDavidson <: EigenSolver end

# Default values
"""
    module KrylovDefaults
        const orth = KrylovKit.ModifiedGramSchmidtIR()
        const krylovdim = Ref(30)
        const maxiter = Ref(100)
        const blockkrylovdim = Ref(100)
        const tol = Ref(1e-12)
        const verbosity = Ref(KrylovKit.WARN_LEVEL)
    end

A module listing the default values for the typical parameters in Krylov based algorithms:

  - `orth = ModifiedGramSchmidtIR()`: the orthogonalization routine used to orthogonalize
    the Krylov basis in the `Lanczos` or `Arnoldi` iteration
  - `krylovdim = 30`: the maximal dimension of the Krylov subspace that will be constructed
  - `maxiter = 100`: the maximal number of outer iterations, i.e. the maximum number of
    times the Krylov subspace may be rebuilt
  - `blockkrylovdim = 100`: the maximal dimension of the Krylov subspace that will be constructed for `BlockLanczos`
  - `tol = 1e-12`: the tolerance to which the problem must be solved, based on a suitable
    error measure, e.g. the norm of some residual.

!!! warning

    The default value of `tol` is a `Float64` value, if you solve problems in `Float32` or
    `ComplexF32` arithmetic, you should always specify a new `tol` as the default value
    will not be attainable.
"""
module KrylovDefaults
using ..KrylovKit
const orth = KrylovKit.ModifiedGramSchmidt2() # conservative choice
const krylovdim = Ref(30)
const maxiter = Ref(100)
const blockkrylovdim = Ref(100)
const tol = Ref(1.0e-12)
const verbosity = Ref(KrylovKit.WARN_LEVEL)
end
