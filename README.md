# KrylovKit.jl

A Julia package collecting a number of Krylov-based algorithms for linear problems, singular
value and eigenvalue problems and the application of functions of linear maps or operators
to vectors.

[![Build Status](https://travis-ci.org/Jutho/KrylovKit.jl.svg?branch=master)](https://travis-ci.org/jutho/KrylovKit.jl)
[![Coverage Status](https://coveralls.io/repos/github/Jutho/KrylovKit.jl/badge.svg?branch=master)](https://coveralls.io/github/Jutho/KrylovKit.jl?branch=master)
[![codecov.io](http://codecov.io/github/Jutho/KrylovKit.jl/coverage.svg?branch=master)](http://codecov.io/github/jutho/KrylovKit.jl?branch=master)

## Overview
`KrylovKit.jl` accepts general functions or callable objects as linear maps, and general Julia
objects with vector like behavior (see below) as vectors.

The high level interface of KrylovKit is provided by the following functions:
*   [`linsolve`]: solve linear systems
*   [`eigsolve`]: find a few eigenvalues and corresponding eigenvectors
*   [`svdsolve`]: find a few singular values and corresponding left and right singular vectors
*   [`exponentiate`]: apply the exponential of a linear map to a vector

## Installation
`KrylovKit.jl` runs on Julia 0.7 or 1.0 and can be installed with
```julia
Pkg.add("KrylovKit.jl")
```
once it will be registered.

## Getting started

Read the documentation: [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://Jutho.github.io/KrylovKit.jl/latest)
