# Functions of matrices and linear operators
Applying a function of a matrix or linear operator to a given vector can in some cases also be
computed using Krylov methods. One example is the inverse function, which exactly
corresponds to what `linsolve` computes: ``A^{-1} * b``. There are other functions ``f``
for which ``f(A) * b`` can be computed using Krylov techniques, i.e. where ``f(A) * b`` can
be well approximated in the Krylov subspace spanned by ``{b, A * b, A^2 * b, ...}``.

Currently, the only family of functions of a linear map for which such a method is
available are the `ϕⱼ(z)` functions which generalize the exponential function
`ϕ₀(z) = exp(z)` and arise in the context of linear non-homogeneous ODEs. The corresponding
Krylov method for computing is an exponential integrator, and is thus available under the
name `expintegrator`. For a linear homogeneous ODE, the solution is a pure exponential, and
the special wrapper `exponentiate` is available:

```@docs
exponentiate
expintegrator
```
