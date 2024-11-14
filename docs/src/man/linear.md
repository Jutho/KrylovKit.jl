# Linear problems

## Linear systems

Linear systems are of the form `A*x=b` where `A` should be a linear map that has the same
type of output as input, i.e. the solution `x` should be of the same type as the right hand
side `b`. They can be solved using the function `linsolve`:

```@docs
linsolve
```

## Automatic differentation

The `linsolve` routine can be used in conjunction with reverse-mode automatic differentiation,
using AD engines that are compatible with the [ChainRules](https://juliadiff.org/ChainRulesCore.jl/dev/)
ecosystem. The adjoint problem of a linear problem is again a linear problem, that requires the
adjoint[^1] of the linear map. If the linear map is an `AbstractMatrix` instance, its `adjoint`
will be used in the `rrule`. If the linear map is implemented as a function `f`, then the AD engine
itself is used to compute the corresponding adjoint via `ChainRulesCore.rrule_via_ad(config, f, x)`.
The specific base point `x` at which this adjoint is computed should not affect the result if `f`
properly represents a linear map. Furthermore, the `linsolve` output is only affected by the linear
map argument and the right hand side argument `b` (from a theoretical perspective, the starting vector
and algorithm parameters should have no effect), so that these two arguments are where the adjoint 
variables need to be propagated to and have a nonzero effect.

The adjoint linear problem (also referred to as cotangent problem) is by default solved using the
same algorithms as the primal problem. However, the `rrule` can be customized to use a different
Krylov algorithm, by specifying the `alg_rrule` keyword argument. Its value can take any of the values
as the `algorithm` argument in `linsolve`.

[^1]: For a linear map, the adjoint or pullback required in the reverse-order chain rule coincides
with its (conjugate) transpose, at least with respect to the standard Euclidean inner product.