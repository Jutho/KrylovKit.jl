# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.

| **Documentation** | **Build Status** | **Digital Object Idenitifier** | **License** |
|:-----------------:|:----------------:|:---------------:|:-----------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][aqua-img]][aqua-url] [![CI][github-img]][github-url] [![][codecov-img]][codecov-url] | [![DOI][doi-img]][doi-url] | [![license][license-img]][license-url] |

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

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.10622234.svg
[doi-url]: https://doi.org/10.5281/zenodo.10622234

## Release notes for the latest version

### v0.9
KrylovKit v0.9 adds to new sets of functionality:
* The function `lssolve` can be used to solve linear least squares problems, i.e. problems of the form `x = argmin(norm(A*x - b))` 
  for a given linear map `A` and vector `b`. Currently, only one algorithm is implemented, namely the LSMR algorithm
  of Fong and Saunders.
* There are now two new functions `reallinsolve` and `realeigsolve`, which are useful when using vectors with complex arithmetic,
  but where the linear map (implemented as a function `f`) acts as a real linear map, meaning that it only satisfies
  `f(α*x) = α*f(x)` when `α` is a real number. This occurs for example when computing the Jacobian of a complex function that is
  not holomorphic, e.g. in the context of automatic differentation. This is implemented by simply wrapping the vector as `RealVec`,
  which is a specific `InnerProductVec` type where the redefined inner product forgets about the imaginary part of the original
  `inner` function, thereby effectively treating the vector as living in a real vector space. Furthermore, in this setting, only
  real linear combinations of vectors are allowed, so that for the case of `eigsolve`, only real eigenvalues and eigenvectors are
  computed. An error will be thrown if the requested list of eigenvalues contains complex eigenvalues. 

## Overview
KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (as defined in the docs) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   `linsolve`: solve linear systems
*   `lssolve`: solve least squares problems
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

The package is tested against Julia `1.6`, the long-term stable release (1.10), the current stable release as well
as nightly builds of the Julia `master` branch on Linux, macOS, and Windows 64-bit architecture and with `1` and `4` threads.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

[issues-url]: https://github.com/Jutho/KrylovKit.jl/issues
