# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.

## Overview
KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (see below) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   [`linsolve`](@ref): solve linear systems `A*x = b`
*   [`eigsolve`](@ref): find a few eigenvalues and corresponding eigenvectors of an
    eigenvalue problem `A*x = λ x`
*   [`geneigsolve`](@ref): find a few eigenvalues and corresponding vectors of a
    generalized eigenvalue problem `A*x = λ*B*x`
*   [`svdsolve`](@ref): find a few singular values and corresponding left and right
    singular vectors `A*x = σ * y` and `A'*y = σ*x`
*   [`exponentiate`](@ref): apply the exponential of a linear map to a vector `x=exp(t*A)*x₀`
*   [`expintegrator`](@ref): exponential integrator for a linear non-homogeneous ODE
    (generalization of `exponentiate`)

Furthermore, for specialised use cases, there are functions that can deal with so-called
"real linear maps", which arise e.g. in the context of differentiable programming:
*   [`reallinsolve`](@ref) and [`realeigsolve`](@ref)


## Package features and alternatives
This section could also be titled "Why did I create KrylovKit.jl"?

There are already a fair number of packages with Krylov-based or other iterative methods, such as
*   [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl): part of the
    [JuliaMath](https://github.com/JuliaMath) organisation, solves linear systems and least
    square problems, eigenvalue and singular value problems
*   [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl): part of the
    [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organisation, solves
    linear systems and least square problems on CPU or GPU for any data type that supports `mul!()`,
    including dense and sparse matrices, and abstract operators such as those defined from
    [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) or
    [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl).
*   [KrylovMethods.jl](https://github.com/lruthotto/KrylovMethods.jl): specific for sparse
    matrices
*   [Expokit.jl](https://github.com/acroy/Expokit.jl): application of the matrix
    exponential to a vector
*   [ArnoldiMethod.jl](https://github.com/haampie/ArnoldiMethod.jl): Implicitly restarted
    Arnoldi method for eigenvalues of a general matrix
*   [JacobiDavidson.jl](https://github.com/haampie/JacobiDavidson.jl): Jacobi-Davidson
    method for eigenvalues of a general matrix
*   [ExponentialUtilities.jl](https://github.com/JuliaDiffEq/ExponentialUtilities.jl): Krylov
    subspace methods for matrix exponentials and `phiv` exponential integrator products. It
    has specialized methods for subspace caching, time stepping, and error testing which are
    essential for use in high order exponential integrators.
*   [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl):
    contains implementations of [high order exponential integrators](https://docs.juliadiffeq.org/latest/solvers/split_ode_solve/#OrdinaryDiffEq.jl-2)
    with adaptive Krylov-subspace calculations for solving semilinear and nonlinear ODEs.

Some of these packages have certainly inspired and influenced the development of KrylovKit.jl.
However, KrylovKit.jl distinguishes itself from the previous packages in the following ways:

1.  KrylovKit accepts general functions to represent the linear map or operator that defines
    the problem, without having to wrap them in a
    [`LinearMap`](https://github.com/Jutho/LinearMaps.jl) or
    [`LinearOperator`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) type.
    Of course, subtypes of `AbstractMatrix` are also supported. If the linear map (always
    the first argument) is a subtype of `AbstractMatrix`, matrix vector multiplication is
    used, otherwise it is applied as a function call.

2.  KrylovKit does not assume that the vectors involved in the problem are actual subtypes
    of `AbstractVector`. Any Julia object that behaves as a vector is supported, so in
    particular higher-dimensional arrays or any custom user type that supports the
    interface as defined in [`VectorInterface.jl`](https://github.com/Jutho/VectorInterface.jl).
    Aside from arrays filled with scalar entries, this includes tuples, named tuples, and
    arbitrarily nested combinations of tuples and arrays. Furthermore, `CuArray` objects
    are fully supported as vectors, so that the application of the linear operator on the
    vector can be executed on a GPU. The computations performed within the Krylov subspace,
    such as diagonalising the projected matrix, are however always performed on the CPU.
    
3.  Since version 0.8, KrylovKit.jl supports reverse-mode AD by defining `ChainRulesCore.rrule` 
    definitions for the most common functionality (`linsolve`, `eigsolve`, `svdsolve`).
    Hence, reverse mode AD engines that are compatible with the [ChainRules](https://juliadiff.org/ChainRulesCore.jl/dev/)
    ecosystem will be able to benefit from an optimized implementation of the adjoint
    of these functions. The `rrule` definitions for the remaining functionality 
    (`geneigsolve` and `expintegrator`, of which `exponentiate` is a special case) will be
    added at a later stage. There is a dedicated documentation page on how to configure these
    `rrule`s, as they also require to solve large-scale linear or eigenvalue problems.

## Current functionality

The following algorithms are currently implemented
*   `linsolve`: [`CG`](@ref), [`GMRES`](@ref), [`BiCGStab`](@ref)
*   `eigsolve`: a Krylov-Schur algorithm (i.e. with tick restarts) for extremal eigenvalues
    of normal (i.e. not generalized) eigenvalue problems, corresponding to
    [`Lanczos`](@ref) for real symmetric or complex hermitian linear maps, and to
    [`Arnoldi`](@ref) for general linear maps.
*   `geneigsolve`: an customized implementation of the inverse-free algorithm of Golub and
    Ye for symmetric / hermitian generalized eigenvalue problems with positive definite
    matrix `B` in the right hand side of the generalized eigenvalue problem ``A v = B v λ``.
    The Matlab implementation was described by Money and Ye and is known as `EIGIFP`; in
    particular it extends the Krylov subspace with a vector corresponding to the step
    between the current and previous estimate, analogous to the locally optimal
    preconditioned conjugate gradient method (LOPCG). In particular, with Krylov dimension
    2, it becomes equivalent to the latter.
*   `svdsolve`: finding largest singular values based on Golub-Kahan-Lanczos
    bidiagonalization (see [`GKL`](@ref))
*   `exponentiate`: a [`Lanczos`](@ref) or [`Arnoldi`](@ref) based algorithm for the action
    of the exponential of linear map.
*   `expintegrator`: [exponential integrator](https://en.wikipedia.org/wiki/Exponential_integrator)
    for a linear non-homogeneous ODE, computes a linear combination of the `ϕⱼ` functions which generalize `ϕ₀(z) = exp(z)`.

## Future functionality?

Here follows a wish list / to-do list for the future. Any help is welcomed and appreciated.

*   More algorithms, including biorthogonal methods:
    -   for `linsolve`: L-GMRES, MINRES, BiCG, IDR(s), ...
    -   for `eigsolve`: BiLanczos, Jacobi-Davidson JDQR/JDQZ, subspace iteration (?), ...
    -   for `geneigsolve`: trace minimization, ...
*   Support both in-place / mutating and out-of-place functions as linear maps
*   Reuse memory for storing vectors when restarting algorithms (related to previous)
*   Support non-BLAS scalar types using GeneralLinearAlgebra.jl and GeneralSchur.jl
*   Least square problems
*   Nonlinear eigenvalue problems
*   Preconditioners
*   Refined Ritz vectors, Harmonic Ritz values and vectors
*   Block versions of the algorithms
*   More relevant matrix functions

Partially done:
*   Improved efficiency for the specific case where `x` is `Vector` (i.e. BLAS level 2
    operations): any vector `v::AbstractArray` which has `IndexStyle(v) == IndexLinear()`
    now benefits from a multithreaded (use `export JULIA_NUM_THREADS = x` with `x` the
    number of threads you want to use) implementation that resembles BLAS level 2 style for
    the vector operations, provided `ClassicalGramSchmidt()`, `ClassicalGramSchmidt2()` or
    `ClassicalGramSchmidtIR()` is chosen as orthogonalization routine.
