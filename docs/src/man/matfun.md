# Functions of matrices and linear maps
Currently, the only family of functions of a linear map for which a method is available are
the so-called `ϕⱼ(z)` functions which generalize the exponential function `ϕ₀(z) = exp(z)`.
These functions arise in the context of linear non-homogeneous ODEs, and the corresponding
Krylov method is an exponentional integrator, hence `expintegrator`. for a linear
homogeneous ODE, the solution is a pure exponential, and the special wrapper `exponentiate`
is available:

```@docs
exponentiate
expintegrator
```
