# Real linear maps

A map $$f: V \to V$$ from some vector space $$V$$ to itself is said to be a real linear map if
it satisfies $$f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$$ for all $$x, y \in V$$ and
all $$\alpha, \beta \in \mathbb{R}$$. When $$V$$ is itself a real vector space, this is just
the natural concept of a linear map. However, this definition can be used even if $$x$$ and
$$y$$ are naturally represented using complex numbers and arithmetic and also admit complex linear
combinations, i.e. if $$V$$ is a complex vector space.

Such real linear maps arise whenever `f(x)` involves calling `conj(x)`, and are for example
obtained in the context of Jacobians (pullbacks) of complex valued functions that are not
holomorphic.

To deal with real linear maps, one should reinterpret $$V$$ as a real vector space, by
restricting the possible linear combinations to those with real scalar coefficients, and by
using the real part of the inner product. When the vectors are explictly represented as
some `AbstractVector{Complex{T}}`, this could be obtained by explicitly splitting
them in their real and imaginary parts and stacking those into `AbstractVector{T}` objects
with twice the original length.

However, KrylovKit.jl admits a different approach, where the original representation of
vectors is kept, and the inner product is simply replaced by its real part. KrylovKit.jl
offers specific methods for solving linear systems and eigenvalue systems in this way. For
linear problems, this is implemented using `reallinsolve`:

```@docs
reallinsolve
```

In the case of eigenvalue systems, a similar method `realeigsolve` is available. In this
context, only real eigenvalues are meaningful, as the corresponding eigenvectors should be
built from real linear combinations of the vectors that span the (real) Krylov subspace.
This approach can also be applied to linear maps on vectors that were naturally real to
begin with, if it is guaranteed that the targetted eigenvalues are real. In that case, also
the associated eigenvectors will be returned using only real arithmic. This is contrast
with `eigsolve`, which will always turn to complex arithmetic if the linear map is real but
not symmetric. An error will be thrown if complex eigenvalues are encountered within the
targetted set.

```@docs
realeigsolve
```

Note that both `reallinsolve` and `realeigsolve` currently only exist with the "expert" mode
interface, where the user has to manually specify the underlying Krylov algorithm and its
parameters, i.e. `GMRES` or `BiCGStab` for `reallinsolve` and `Arnoldi` for `realeigsolve`.