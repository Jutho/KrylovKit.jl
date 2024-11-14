# Singular value problems

## Singular values and singular vectors
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

Both `svdsolve` and the adjoint problem associated with it require the action of the linear map as
well as of its adjoint[^1]. Hence, no new information about the linear map is required for the adjoint
problem. However, the linear map is the only argument that affects the `svdsolve` output (from a
theoretical perspective, the starting vector and algorithm parameters should have no effect), so that
this is where the adjoint variables need to be propagated to.

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
