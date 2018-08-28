# KrylovKit.jl Documention

`KrylovKit.jl` is a Julia package that collects a number of Krylov-based algorithms for linear
problems, singular value and eigenvalue problems and the application of functions of linear
maps or operators to vectors.

## Contens

```@contents
```

## Defining features
There are a number of packages with Krylov-based or other iterative methods, such as
*   [`IterativeSolvers.jl`](https://github.com/JuliaMath/IterativeSolvers.jl): part of the
    [`JuliaMath`](https://github.com/JuliaMath) organisation, solves linear systems and least
    square problems, eigenvalue and singular value problems
*   [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl): part of the
    [`JuliaSmoothOptimizers`](https://github.com/JuliaSmoothOptimizers) organisation, solves
    linear systems and least square problems, specific for linear operators from
    [`LinearOperators.jl`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
*   [`KrylovMethods.jl`](https://github.com/lruthotto/KrylovMethods.jl): specific for sparse matrices
*   [`Expokit.jl`](https://github.com/acroy/Expokit.jl): application of the matrix exponential to a vector

`KrylovKit.jl` distinguishes itself from the previous packages the following two ways

1.  `KrylovKit` accepts general functions `f` to represent the linear map or operator that defines
    the problem, without having to wrap them in a [`LinearMap`](https://github.com/Jutho/LinearMaps.jl)
    or [`LinearOperator`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl) type.
    Of course, subtypes of `AbstractMatrix` are also supported. The linear map is always the
    first argument in `linsolve(f,...)`, `eigsolve(f,...)`, `svdsolve(f,...)`, `exponentiate(f,...)`
    so that Julia's `do` block construction can be used, e.g.
    ```julia
    linsolve(...) do x
        # some linear operation on x
    end
    ```
    If the first argument `f isa AbstractMatrix`, matrix vector multiplication is used, otherwise
    `f` is called as a function.

2.  `KrylovKit` does not assume that the vectors involved in the problem are actual subtypes of
    `AbstractVector`. Any Julia object that behaves as a vector (in the way defined below) is
    supported, so in particular higher-dimensional arrays or any custom user type that supports
    the following functions (with `v` and `w` two instances of this type and `α` a scalar (`Number`)):
    *   `Base.eltype(v)`: the scalar type (i.e. `<:Number`) of the data in `v`
    *   `Base.similar(v, [T::Type<:Number])`: a way to construct additional similar vectors,
        possibly with a different scalar type `T`.
    *   `Base.copyto!(w, v)`: copy the contents of `v` to a preallocated vector `w`
    *   `Base.fill!(w, α)`: fill all the scalar entries of `w` with value `α`; this is only
        used in combination with `α = 0` to create a zero vector. Note that `Base.zero(v)` does
        not work for this purpose if we want to change the scalar `eltype`. We can also not
        use `rmul!(v, 0)` (see below), since `NaN*0` yields `NaN`.
    *   `LinearAlgebra.mul!(w, v, α)`: out of place scalar multiplication; multiply
        vector `v` with scalar `α` and store the result in `w`
    *   `LinearAlgebra.rmul!(v, α)`: in-place scalar multiplication of `v` with `α`.
    *   `LinearAlgebra.axpy!(α, v, w)`: store in `w` the result of `α*v + w`
    *   `LinearAlgebra.axpby!(α, v, β, w)`: store in `w` the result of `α*v + β*w`
    *   `LinearAlgebra.dot(v,w)`: compute the inner product of two vectors
    *   `LinearAlgebra.norm(v)`: compute the 2-norm of a vector

    In particular, `KrylovKit` provides two types satisfying the above requirements that might
    facilitate certain applications:
    * [`RecursiveVec`](@ref) can be used for grouping a set of vectors into a single vector like
    structure (can be used recursively). The reason that e.g. `Vector{<:Vector}` cannot be used
    for this is that it returns the wrong `eltype` and methods like `similar(v, T)` and `fill!(v, α)`
    don't work correctly.
    * [`InnerProductVec`](@ref) can be used to redefine the inner product (i.e. `dot`) and corresponding
    norm (`norm`) of an already existing vector like object. The latter should help with implementing
    certain type of preconditioners and solving generalized eigenvalue problems with a positive
    definite matrix in the right hand side.


## Current functionality

The following algorithms are currently implemented
*   `orthogonalization`: Classical & Modified Gram Schmidt, possibly with a second round or
    an adaptive number of rounds of reorthogonalization.
*   `linsolve`: [`GMRES`](@ref) without preconditioning
*   `eigsolve`: a Krylov-Schur algorithm (i.e. with tick restarts) for extremal eigenvalues of
    normal (i.e. not generalized) eigenvalue problems, corresponding to [`Lanczos`](@ref) for
    real symmetric or complex hermitian linear maps, and to [`Arnoldi`](@ref) for general linear maps.
*   `svdsolve`: finding largest singular values by using `eigsolve` of the circulant matrix with
    the `Lanczos` algorithm.
*   `exponentiate`: a [`Lanczos`](@ref) based algorithm for the action of the exponential of
    a real symmetric or complex hermitian linear map.

## Future functionality?

Below is a wish list / to-do list for the future. Any help is welcomed and appreciated.

*   More algorithms, including biorthogonal methods:
    -   for `linsolve`: CG, MINRES, BiCG, BiCGStab, ...
    -   for `eigsolve`: BiLanczos, Jacobi-Davidson (?), subspace iteration (?), ...
    -   for `svdsolve`: Golub-Kahan-Lanczos
    -   for `exponentiate`: Arnoldi (currently only Lanczos supported)
*   Generalized eigenvalue problems: LOPCG, EIGFP and trace minimization
*   Least square problems
*   Nonlinear eigenvalue problems
*   Preconditioners
*   Harmonic ritz values
*   Support both in-place / mutating and out-of-place functions as linear maps
*   Reuse memory for storing vectors when restarting algorithms
*   Improved efficiency for the specific case where `x` is `Vector` (i.e. BLAS level 2 operations)
*   Block versions of the algorithms
*   More relevant matrix functions
