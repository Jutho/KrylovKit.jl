# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.


| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] |

## Release notes for the latest version

### v0.3
*   a new method `expintegrator` that generalizes `exponentiate` (and the latter is now a
    special case of the former), i.e. it computes a linear combination of the so-called
    ``ϕⱼ`` functions that corresponds to the solution of a linear inhomogeneous ODE.

*   stricter handling of keyword arguments; unknown keyword arguments are no longer ignored
    but give an error.

### v0.2
*   a new method `geneigsolve` for generalized eigenvalue problems ``A x = λ B x``, with a
    single implementation for symmetric/hermitian problems with positive definite `B`, based
    on the Golub-Ye inverse free algorithm.

*   a `verbosity` keyword that takes integer values to control the amount of information
    that is printed while the algorithm is running.

### v0.1: first official release

## Overview
KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (as defined in the docs) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   `linsolve`: solve linear systems
*   `eigsolve`: find a few eigenvalues and corresponding eigenvectors
*   `geneigsolve`: find a few generalized eigenvalues and corresponding vectors
*   `svdsolve`: find a few singular values and corresponding left and right singular vectors
*   `exponentiate`: apply the exponential of a linear map to a vector
*   `expintegrator`: exponential integrator for a linear non-homogeneous ODE, computes a
    linear combination of the so-called `ϕⱼ` functions which generalize `ϕ₀(z) = exp(z)`.

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

The package is tested against Julia `0.7`, `1.0`, `1.1` and the nightly builds of the Julia `master` branch on Linux, macOS, and Windows, 32- and 64-bit architecture and with `1` and `4` threads.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.


[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://Jutho.github.io/KrylovKit.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://Jutho.github.io/KrylovKit.jl/stable

[travis-img]: https://travis-ci.org/Jutho/KrylovKit.jl.svg?branch=master
[travis-url]: https://travis-ci.org/Jutho/KrylovKit.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/Jutho/KrylovKit.jl?svg=true&branch=master
[appveyor-url]: https://ci.appveyor.com/project/Jutho/krylovkit-jl/branch/master

[codecov-img]: https://codecov.io/gh/Jutho/KrylovKit.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/KrylovKit.jl

[coveralls-img]: https://coveralls.io/repos/github/Jutho/KrylovKit.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/Jutho/KrylovKit.jl

[issues-url]: https://github.com/Jutho/KrylovKit.jl/issues
