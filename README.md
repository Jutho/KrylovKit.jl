# KrylovKit.jl

A package collecting a number of Krylov-based algorithms for linear problems, eigenvalue problems and the application of functions of linear maps or operators to vectors.

[![Build Status](https://travis-ci.org/jutho/KrylovKit.jl.svg?branch=master)](https://travis-ci.org/jutho/KrylovKit.jl)
[![Coverage Status](https://coveralls.io/repos/jutho/KrylovKit.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/jutho/KrylovKit.jl?branch=master)
[![codecov.io](http://codecov.io/github/jutho/KrylovKit.jl/coverage.svg?branch=master)](http://codecov.io/github/jutho/KrylovKit.jl?branch=master)

--------------------------------------------------------------------------------
So far, the starting point into KrylovKit are the general purpose functions [`linsolve`](ref), [`eigsolve`](@ref) and [`exponentiate`](@ref). KrylovKit distinguishes itself from similar packages in the Julia ecosystem in the following two ways

1.  `KrylovKit` accepts general functions `f` to represent the linear map or operator that defines the problem, but does of course also accept subtypes of `AbstractMatrix`. The linear map is always the first argument in `linsolve(f,...)`, `eigsolve(f,...)`, `exponentiate(f,...)` so that Julia's `do` block construction can be used, e.g.

    ```julia
    linsolve(...) do x
    # some linear operation on x
    end
    ```

    If the first argument `f isa AbstractMatrix`, matrix vector multiplication is used, otherwise `f` is called as a function.

2.  `KrylovKit` does not assume that the vectors involved in the problem are actual subtypes of `AbstractVector`. Any Julia object that behaves as a vector (in the way defined below) is supported, so in particular higher-dimensional arrays or any custom user type that supports the following functions (with `v` and `w` two instances of this type and `α` a scalar (`Number`)):
    -   `Base.eltype(v)`: the scalar type (i.e. `<:Number`) of the data in `v`
    -   `Base.similar(v, [T::Type<:Number])`: a way to construct additional similar vectors, possibly with a different scalar type `T`.
    -   `Base.copyto!(w, v)`: copy the contents of `v` to a preallocated vector `w`
    -   `Base.fill!(w, α)`: fill all the scalar entries of `w` with value `α`; this is only used in combination with `α = 0` to create a zero vector. Note that `Base.zero(v)` does not work for this purpose if we want to change the scalar `eltype`. We can also not use `rmul!(v, 0)` (see below), since `NaN*0` yields `NaN`.
    -   `LinearAlgebra.mul!(w, v, α)`: out of place scalar multiplication; multiply vector `v` with scalar `α` and store the result in `w`
    -   `LinearAlgebra.rmul!(v, α)`: in-place scalar multiplication of `v` with `α`.
    -   `LinearAlgebra.axpy!(α, v, w)`: store in `w` the result of `α*v + w`
    -   `LinearAlgebra.axpby!(α, v, β, w)`: store in `w` the result of `α*v + β*w`
    -   `LinearAlgebra.dot(v,w)`: compute the inner product of two vectors
    -   `LinearAlgebra.norm(v)`: compute the 2-norm of a vector

Currently implemented are the following algorithms:

-   `orthogonalization`: Classical & Modified Gram Schmidt, possibly with a second round or an adaptive number of rounds of reorthogonalization.
-   `linsolve`: [`GMRES`](@ref) without preconditioning
-   `eigsolve`: a Krylov-Schur algorithm (i.e. with tick restarts) for extremal eigenvalues of normal (i.e. not generalized) eigenvalue problems, corresponding to [`Lanczos`](@ref) for real symmetric or complex hermitian linear maps, and to [`Arnoldi`](@ref) for general linear maps.
-   `exponentiate`: a [`Lanczos`](@ref) based algorithm for the action of the exponential of a real symmetric or complex hermitian linear map.

Furthermore, `KrylovKit` provides two vector like types that might prove useful in certain applications.
*   [`RecursiveVec`](@ref) can be used for grouping a set of vectors into a single vector like structure (can be used recursively). The reason that e.g. `Vector{<:Vector}` cannot be used for this is that it returns the wrong `eltype` and methods like `similar(v, T)` and `fill!(v, α)` don't work correctly.
*   [`InnerProductVec`](@ref) can be used to redefine the inner product (i.e. `dot`) and corresponding norm (`norm`) of an already existing vector like object. The latter should help with implementing certain type of preconditioners and solving generalized eigenvalue problems with a positive definite matrix in the right hand side.

Some TODO list of possible features for the future:
-   More linear solvers: conjugate gradient
-   Biorthogonal methods for linear problems and eigenvalue problems: BiCG, BiCGStab, BiLanczos
-   Preconditioners
-   Exponentiate for a general (non-hermitian) linear map, based on `Arnoldi`
-   Singular values: `svdsolve`
-   Harmonic ritz values
-   Generalized eigenvalue problems: LOPCG, EIGFP and trace minimization
-   Support both in-place / mutating and out-of-place functions as linear maps
-   Reuse memory for storing vectors when restarting algorithms
-   Improved efficiency for the specific case where `x` is `Vector` (i.e. BLAS level 2 operations)
-   More relevant matrix functions
