# Singular value problems
It is possible to iteratively compute a few singular values and corresponding left and
right singular vectors using the function `svdsolve`:

```@docs
svdsolve
```

## Automatic differentation

The `svdsolve` routine can be used in conjunction with reverse-mode automatic differentiation, 
using AD engines that are compatible with the [ChainRules](https://juliadiff.org/ChainRulesCore.jl/dev/)
ecosystem. The adjoint problem of a singular value problem contains a linear problem, although it
can also be formulated as an eigenvalue problem. Details about this approach will be published in a
forthcoming manuscript.

In either case, the adjoint problem requiers the adjoint[^1] of the linear map. If the linear map is
an `AbstractMatrix` instance, its `adjoint` will be used in the `rrule`. If the linear map is implemented 
as a function `f`, then the AD engine itself is used to compute the corresponding adjoint via 
`ChainRulesCore.rrule_via_ad(config, f, x)`. The specific base point `x` at which this adjoint is
computed should not affect the result if `f` properly represents a linear map.

The adjoint problem (also referred to as cotangent problem) can thus be solved as a linear problem
or as an eigenvalue problem. Note that this eigenvalue problem is never symmetric or Hermitian.
The different implementations of the `rrule` can be selected using the `alg_rrule` keyword argument. 
If a linear solver such as `GMRES` or `BiCGStab` is specified, the adjoint problem requires solving a]
number of linear problems equal to the number of requested singular values and vectors. If an 
eigenvalue solver is specified, for which `Arnoldi` is essentially the only option, then the adjoint
problem is solved as a single (but larger) eigenvalue problem.

Note that the common pair of left and right singular vectors has an arbitrary phase freedom.
Hence, a well-defined cost function constructed from singular should depend on these in such a way 
that its value is not affected by simultaneously changing the left and right singular vector with
a common phase factor, i.e. the cost function should be 'gauge invariant'. If this is not the case, 
the cost function is said to be 'gauge dependent', and this can be detected in the resulting adjoint
variables for those singular vectors. The KrylovKit `rrule` for `svdsolve` will print a warning if
it detects from the incoming adjoint variables that the cost function is gauge dependent. This
warning can be suppressed by passing `alg_rrule` an algorithm with `verbosity=-1`.

[^1]: For a linear map, the adjoint or pullback required in the reverse-order chain rule coincides
with its (conjugate) transpose, at least with respect to the standard Euclidean inner product.
