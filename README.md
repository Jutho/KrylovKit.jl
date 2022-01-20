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

### v0.5
This version introduces (minimal) breaking changes, if you use KrylovKit.jl with custom
vector types: KrylovKit.jl no longer depends on `eltype(::YourCustomVector)` and
`similar(::YourCustumVector, ::Type{<:Number})`. Instead, KrylovKit.jl does now rely on
`Base.:*(::Number, ::YourCustomVector)` to be defined as a means of creating new vectors,
possibly with a different scalar type, so as to be able to represent this computation. Note
that `Base.similar(::YourCustomVector)` (without the second argument) should still be
defined to create uninitialized vectors of the same type as the one of the argument.

The motivation for this is that using `eltype(::YourCustomVector)` to represent its scalar
type, was often not the compatible with the requirements for `Base.eltype` if your type also
supports iteration or indexing.

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
