# Introduction

```@contents
Pages = ["man/intro.md", "man/linear.md", "man/eig.md", "man/svd.md", "man/matfun.md",
"man/algorithms.md", "man/implementation.md"]
Depth = 2
```

## Installing

Install KrylovKit.jl via the package manager:
```julia
using Pkg
Pkg.add("KrylovKit")
```

KrylovKit.jl is a pure Julia package; no dependencies (aside from the Julia standard
library) are required.

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

The for high-level function [`linsolve`](@ref), [`eigsolve`](@ref), [`geneigsolve`](@ref),
[`svdsolve`](@ref), [`exponentiate`](@ref) and [`expintegrator`](@ref) follow a common interface
```julia
results..., info = problemsolver(A, args...; kwargs...)
```
where `problemsolver` is one of the functions above. Here, `A` is the linear map in the
problem, which could be an instance of `AbstractMatrix`, or any function or callable object
that encodes the action of the linear map on a vector. In particular, one can write the
linear map using Julia's `do` block syntax as
```julia
results..., info = problemsolver(args...; kwargs...) do x
    y = # implement linear map on x
    return y
end
```
Read the documentation for problems that require both the linear map and its adjoint to be
implemented, e.g. [`svdsolve`](@ref), or that require two different linear maps, e.g.
[`geneigsolve`](@ref).

Furthermore, `args` is a set of additional arguments to specify the problem. The keyword
arguments `kwargs` contain information about the linear map (`issymmetric`, `ishermitian`,
`isposdef`) and about the solution strategy (`tol`, `krylovdim`, `maxiter`). Finally, there
is a keyword argument `verbosity` that determines how much information is printed to
`STDOUT`. The default value `verbosity = 0` means that no information will be printed. With
`verbosity = 1`, a single message at the end of the algorithm will be displayed, which is a
warning if the algorithm did not succeed in finding the solution, or some information if it
did. For `verbosity = 2`, information about the current state is displayed after every
iteration of the algorithm. Finally, for `verbosity > 2`, information about the individual Krylov expansion steps is displayed.

The return value contains one or more entries that define the solution, and a final
entry `info` of type `ConvergeInfo` that encodes information about the solution, i.e.
whether it has converged, the residual(s) and the norm thereof, the number of operations
used:
```@docs
KrylovKit.ConvergenceInfo
```

There is also an expert interface where the user specifies the algorithm that should be used
explicitly, i.e.
```julia
results..., info = problemsolver(A, args..., algorithm(; kwargs...))
```
Most `algorithm` constructions take the same keyword arguments (`tol`, `krylovdim`,
`maxiter` and `verbosity`) discussed above.

While KrylovKit.jl does currently not provide a general interface for including
preconditioners, it is possible to e.g. use a modified inner product. KrylovKit.jl provides
a specific type for this purpose:
```@docs
InnerProductVec
```
