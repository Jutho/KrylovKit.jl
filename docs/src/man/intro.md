# Introduction

```@contents
Pages = ["man/intro.md", "man/linear.md", "man/eig.md", "man/svd.md", "man/matfun.md", "man/algorithms.md", "man/implementation.md"]
Depth = 2
```

## Installing

Install KrylovKit.jl via the package manager:
```julia
using Pkg
Pkg.add("KrylovKit")
```

KrylovKit.jl is a pure Julia package; no dependencies (aside from the Julia standard library)
are required.

## Getting started

After installation, start by loading `KrylovKit`

```julia
using KrylovKit
```
The help entry of the `KrylovKit` module states
```@docs
KrylovKit
```

## Common interface

The for high-level function [`linsolve`](@ref), [`eigsolve`](@ref), [`svdsolve`](@ref) and [`exponentiate`](@ref)
follow a common interface
```julia
results..., info = problemsolver(A, args...; kwargs...)
```
where `problemsolver` is one of the functions above. Here, `A` is the linear map in the problem,
which could be an instance of `AbstractMatrix`, or any function or callable object that encodes
the action of the linear map on a vector. In particular, one can write the linear map using
Julia's `do` block syntax as
```julia
results..., info = problemsolver(args...; kwargs...) do x
    y = # implement linear map on x
    return y
end
```
Read the documentation for problems that require both the linear map and its adjoint to be
implemented, e.g. [`svdsolve`](@ref).

Furthermore, `args` is a set of additional arguments to specify the problem. The keyword arguments
`kwargs` contain information about the linear map (`issymmetric`, `ishermitian`, `isposdef`) and
about the solution strategy (`tol`, `krylovdim`, `maxiter`). A suitable algorithm for the problem
is then chosen.

The return value contains one or more entries that define the solution, and a final
entry `info` of type `ConvergeInfo` that encodes information about the solution, i.e. wether it
has converged, the residual(s) and the norm thereof, the number of operations used:
```@docs
KrylovKit.ConvergenceInfo
```

There is also an expert interface where the user specifies the algorithm that should be used
explicitly, i.e.
```julia
results..., info = problemsolver(A, args..., algorithm)
```
