# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.

| **Documentation** | **Build Status** | **License** |
|:-----------------:|:----------------:|:-----------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][aqua-img]][aqua-url] [![CI][github-img]][github-url] [![][codecov-img]][codecov-url] | [![license][license-img]][license-url] |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jutho.github.io/KrylovKit.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jutho.github.io/KrylovKit.jl/stable

[github-img]: https://github.com/Jutho/KrylovKit.jl/workflows/CI/badge.svg
[github-url]: https://github.com/Jutho/KrylovKit.jl/actions?query=workflow%3ACI

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[codecov-img]: https://codecov.io/gh/Jutho/KrylovKit.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/KrylovKit.jl

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE.md

## Release notes for the latest version

### v0.7
This version now depends on and uses [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)
to define the vector-like behavior of the input vectors, rather than some minimal set of
methods from `Base` and `LinearAlgebra`. The advantage is that many more types from standard
Julia are now supported out of the box, such as nested vectors or immutable objects such as
tuples. For custom user types for which the old set of required methods was implemented, there
are fallback definitions of the methods in VectorInferace.jl such that these types should still
be supported, but this might result in warnings being printed. It is recommend to implement full
support for at least the methods in VectorInterface without bang or with double bang, where the
latter set of methods can use in-place mutation if your type supports this behavior.

In particular, tuples are now supported:

```julia
julia> values, vectors, info = eigsolve(t -> cumsum(t) .+ 0.5 .* reverse(t), (1,0,0,0));

julia> values
4-element Vector{ComplexF64}:
  2.5298897746721303 + 0.0im
  0.7181879189193713 + 0.4653321688070444im
  0.7181879189193713 - 0.4653321688070444im
 0.03373438748912972 + 0.0im

julia> vectors
4-element Vector{NTuple{4, ComplexF64}}:
 (0.25302539267845964 + 0.0im, 0.322913174072047 + 0.0im, 0.48199234088257203 + 0.0im, 0.774201921982351 + 0.0im)
 (0.08084058845575778 + 0.46550907490257704im, 0.16361072959559492 - 0.20526827902633993im, -0.06286027036719286 - 0.6630573167350086im, -0.47879640378455346 - 0.18713670961291684im)
 (0.08084058845575778 - 0.46550907490257704im, 0.16361072959559492 + 0.20526827902633993im, -0.06286027036719286 + 0.6630573167350086im, -0.47879640378455346 + 0.18713670961291684im)
 (0.22573986355213632 + 0.0im, -0.5730667760748933 + 0.0im, 0.655989711683001 + 0.0im, -0.4362493350466509 + 0.0im)
```

## Overview
KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (as defined in the docs) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   `linsolve`: solve linear systems
*   `eigsolve`: find a few eigenvalues and corresponding eigenvectors
*   `geneigsolve`: find a few generalized eigenvalues and corresponding vectors
*   `svdsolve`: find a few singular values and corresponding left and right singular vectors
*   `exponentiate`: apply the exponential of a linear map to a vector
*   `expintegrator`: [exponential integrator](https://en.wikipedia.org/wiki/Exponential_integrator)
    for a linear non-homogeneous ODE, computes a linear combination of the `ϕⱼ` functions which generalize `ϕ₀(z) = exp(z)`.

## Installation
`KrylovKit.jl` can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
pkg> add KrylovKit
```

Or, equivalently, via the `Pkg` API:
```julia
julia> import Pkg; Pkg.add("KrylovKit.jl")
```

## Documentation

-   [**STABLE**][docs-stable-url] - **documentation of the most recently tagged version.**
-   [**DEVEL**][docs-dev-url] - *documentation of the in-development version.*

## Project Status

The package is tested against Julia `1.0`, the current stable and the nightly builds of the Julia `master` branch on Linux, macOS, and Windows, 32- and 64-bit architecture and with `1` and `4` threads.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

[issues-url]: https://github.com/Jutho/KrylovKit.jl/issues
