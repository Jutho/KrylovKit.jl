# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.

[![Build Status](https://travis-ci.org/Jutho/KrylovKit.jl.svg?branch=master)](https://travis-ci.org/jutho/KrylovKit.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![codecov.io](http://codecov.io/github/Jutho/KrylovKit.jl/coverage.svg?branch=master)](http://codecov.io/github/jutho/KrylovKit.jl?branch=master)

## Overview
KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (as defined in the docs) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   `linsolve`: solve linear systems
*   `eigsolve`: find a few eigenvalues and corresponding eigenvectors
*   `svdsolve`: find a few singular values and corresponding left and right singular vectors
*   `exponentiate`: apply the exponential of a linear map to a vector

## Installation
`KrylovKit.jl` runs on Julia 0.7 or 1.0 and can be installed by entering the package REPL mode
(i.e. typing `]`) and then
```
pkg> add KrylovKit
```
or directly in the Julia REPL
```julia
julia> using Pkg
julia> Pkg.add("KrylovKit.jl")
```

## Getting started

Read the documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Jutho.github.io/KrylovKit.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://Jutho.github.io/KrylovKit.jl/latest)
