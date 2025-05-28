"""
    function exponentiate(A, t::Number, x; kwargs...)
    function exponentiate(A, t::Number, x, algorithm)

Compute ``y = exp(t*A) x``, where `A` is a general linear map, i.e. a `AbstractMatrix` or
just a general function or callable object and `x` is of any Julia type with vector like
behavior.

### Arguments:

The linear map `A` can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object that implements the action of the linear map on a vector. If `A` is an
`AbstractMatrix`, `x` is expected to be an `AbstractVector`, otherwise `x` can be of any
type that behaves as a vector and supports the required methods (see KrylovKit docs).

The time parameter `t` can be real or complex, and it is better to choose `t` e.g. imaginary
and `A` hermitian, then to absorb the imaginary unit in an antihermitian `A`. For the
former, the Lanczos scheme is used to built a Krylov subspace, in which an approximation to
the exponential action of the linear map is obtained. The argument `x` can be of any type
and should be in the domain of `A`.

### Return values:

The return value is always of the form `y, info = exponentiate(...)` with

  - `y`: the result of the computation, i.e. `y = exp(t*A)*x`

  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: 0 or 1 if the solution `y` at time `t` was found with an
        error below the requested tolerance per unit time, i.e. if `info.normres <= tol * abs(t)`
      + `info.residual::Nothing`: value `nothing`, there is no concept of a residual in
        this case
      + `info.normres::Real`: a (rough) estimate of the total error accumulated in the
        solution
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
      + `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

!!! warning "Check for convergence"

    By default (i.e. if `verbosity = SILENT_LEVEL`, see below), no warning is printed if the solution
    was not found with the requested precision, so be sure to check `info.converged == 1`.

### Keyword arguments:

Keyword arguments and their default values are given by:

  - `verbosity::Int = SILENT_LEVEL`: verbosity level, i.e. 
    - SILENT_LEVEL (suppress all messages)
    - WARN_LEVEL (only warnings)
    - STARTSTOP_LEVEL (one message with convergence info at the end)
    - EACHITERATION_LEVEL (progress info after every iteration)
    - EACHITERATION_LEVEL+ (all of the above and additional information about the Lanczos or Arnoldi iteration)
  - `krylovdim = 30`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should
    not necessarily support the `Base.length` function. If you know the actual problem
    dimension is smaller than the default value, it is useful to reduce the value of
    `krylovdim`, though in principle this should be detected.
  - `tol = 1e-12`: the requested accuracy per unit time, i.e. if you want a certain
    precision `ϵ` on the final result, set `tol = ϵ/abs(t)`. If you work in e.g. single
    precision (`Float32`), you should definitely change the default value.
  - `maxiter::Int = 100`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
  - `issymmetric`: if the linear map is symmetric, only meaningful if `T<:Real`
  - `ishermitian`: if the linear map is hermitian
    The default value for the last two depends on the method. If an `AbstractMatrix` is
    used, `issymmetric` and `ishermitian` are checked for that matrix, otherwise the default
    values are `issymmetric = false` and `ishermitian = T <: Real && issymmetric`.
  - `eager::Bool = false`: if true, eagerly try to compute the result after every expansion
    of the Krylov subspace to test for convergence, otherwise wait until the Krylov subspace
    as dimension `krylovdim`. This can result in a faster return, for example if the total
    time for the evolution is quite small, but also has some overhead, as more computations
    are performed after every expansion step.

### Algorithm

This is actually a simple wrapper over more general method [`expintegrator`](@ref) for
for integrating a linear non-homogeneous ODE.
"""
function exponentiate end

exponentiate(A, t::Number, v; kwargs...) = expintegrator(A, t, v; kwargs...)
exponentiate(A, t::Number, v, alg::Union{Lanczos,Arnoldi}) = expintegrator(A, t, (v,), alg)
