# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.


| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] |

## Release notes for the latest version

*   a new method `geneigsolve` for generalized eigenvalue problems ``A x = λ B x``, with a
    single implementation for symmetric/hermitian problems with positive definite `B`, based
    on the Golub-Ye inverse free algorithm.

*   a `verbosity` keyword that takes integer values to control the amount of information
    that is printed while the algorithm is running.

## Overview
KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (as defined in the docs) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   `linsolve`: solve linear systems
*   `eigsolve`: find a few eigenvalues and corresponding eigenvectors
*   `geneigsolve`: find a few generalized eigenvalues and corresponding vectors
*   `svdsolve`: find a few singular values and corresponding left and right singular vectors
*   `exponentiate`: apply the exponential of a linear map to a vector

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

The package is tested against Julia `0.7`, `1.0` and the nightly builds of the Julia `master` branch on Linux, macOS, and Windows.

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
