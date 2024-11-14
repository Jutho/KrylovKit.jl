# Eigenvalue problems

## Eigenvalues and eigenvectors
Finding a selection of eigenvalues and corresponding (right) eigenvectors of a linear map
can be accomplished with the `eigsolve` routine:
```@docs
eigsolve
```

Which eigenvalues are targeted can be specified using one of the symbols `:LM`, `:LR`,
`:SR`, `:LI` and `:SI` for largest magnitude, largest and smallest real part, and largest
and smallest imaginary part respectively. Alternatively, one can just specify a general
sorting operation using `EigSorter`
```@docs
EigSorter
```

For a general matrix, eigenvalues and eigenvectors will always be returned with complex
values for reasons of type stability. However, if the linear map and initial guess are
real, most of the computation is actually performed using real arithmetic, as in fact the
first step is to compute an approximate partial Schur factorization. If one is not
interested in the eigenvectors, one can also just compute this partial Schur factorization
using `schursolve`, for which only an 'expert' method call is available
```@docs
schursolve
```
Note that, for symmetric or hermitian linear maps, the eigenvalue and Schur factorization
are equivalent, and one should only use `eigsolve`. There is no `schursolve` using the `Lanczos` algorithm.

Another example of a possible use case of `schursolve` is if the linear map is known to have
a unique eigenvalue of, e.g. largest magnitude. Then, if the linear map is real valued, that
largest magnitude eigenvalue and its corresponding eigenvector are also real valued.
`eigsolve` will automatically return complex valued eigenvectors for reasons of type
stability. However, as the first Schur vector will coincide with the first eigenvector, one
can instead use
```julia
T, vecs, vals, info = schursolve(A, x⁠₀, 1, :LM, Arnoldi(...))
```
and use `vecs[1]` as the real valued eigenvector (after checking `info.converged`)
corresponding to the largest magnitude eigenvalue of `A`.

More generally, if you want to compute several eigenvalues of a real linear map, and you know
that all of them are real, so that also the associated eigenvectors will be real, then you
can use the [`realeigsolve`](@ref) method.

## Automatic differentation

The `eigsolve` (and `realeigsolve`) routine can be used in conjunction with reverse-mode automatic 
differentiation, using AD engines that are compatible with the [ChainRules](https://juliadiff.org/ChainRulesCore.jl/dev/)
ecosystem. The adjoint problem of an eigenvalue problem is a linear problem, although it can also
be formulated as an eigenvalue problem. Details about this approach will be published in a
forthcoming manuscript.

In either case, the adjoint problem requires the adjoint[^1] of the linear map. If the linear map is
an `AbstractMatrix` instance, its `adjoint` will be used in the `rrule`. If the linear map is implemented 
as a function `f`, then the AD engine itself is used to compute the corresponding adjoint via 
`ChainRulesCore.rrule_via_ad(config, f, x)`. The specific base point `x` at which this adjoint is
computed should not affect the result if `f` properly represents a linear map. Furthermore, the linear
map is the only argument that affects the `eigsolve` output (from a theoretical perspective, the
starting vector and algorithm parameters should have no effect), so that this is where the adjoint 
variables need to be propagated to and have a nonzero effect.

The adjoint problem (also referred to as cotangent problem) can thus be solved as a linear problem
or as an eigenvalue problem. Note that this eigenvalue problem is never symmetric or Hermitian,
even if the primal problem is. The different implementations of the `rrule` can be selected using
the `alg_rrule` keyword argument. If a linear solver such as `GMRES` or `BiCGStab` is specified,
the adjoint problem requires solving a number of linear problems equal to the number of requested
eigenvalues and eigenvectors. If an eigenvalue solver is specified, for which `Arnoldi` is essentially
the only option, then the adjoint problem is solved as a single (but larger) eigenvalue problem.

Note that the phase of an eigenvector is not uniquely determined. Hence, a well-defined cost function
constructed from eigenvectors should depend on these in such a way that its value is not affected
by changing the phase of those eigenvectors, i.e. the cost function should be 'gauge invariant'.
If this is not the case, the cost function is said to be 'gauge dependent', and this can be detected
in the resulting adjoint variables for those eigenvectors. The KrylovKit `rrule` for `eigsolve`
will print a warning if it detects from the incoming adjoint variables that the cost function is gauge
dependent. This warning can be suppressed by passing `alg_rrule` an algorithm with `verbosity=-1`.

## Generalized eigenvalue problems

Generalized eigenvalues `λ` and corresponding vectors `x` of the generalized eigenvalue
problem ``A x = λ B x`` can be obtained using the method `geneigsolve`. Currently, there is
only one algorithm, which does not require inverses of `A` or `B`, but is restricted to
symmetric or hermitian generalized eigenvalue problems where the matrix or linear map `B`
is positive definite. Note that this is not reflected in the default values for the keyword
arguments `issymmetric`, `ishermitian` and `isposdef`, so that these should be set
explicitly in order to comply with this restriction. If `A` and `B` are actual instances of
`AbstractMatrix`, the default value for the keyword arguments will try to check these
properties explicitly.

```@docs
geneigsolve
```

Currently, there is `rrule` and thus no automatic differentiation support for `geneigsolve`.

[^1]: For a linear map, the adjoint or pullback required in the reverse-order chain rule coincides
with its (conjugate) transpose, at least with respect to the standard Euclidean inner product.
