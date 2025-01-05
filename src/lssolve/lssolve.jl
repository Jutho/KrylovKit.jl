"""
    lssolve(A::AbstractMatrix, b::AbstractVector, [λ = 0]; kwargs...)
    lssolve(f, b, [λ = 0]; kwargs...)
    # expert version:
    lssolve(f, b, algorithm, [λ = 0])

Compute a least squares solution `x` to the problem `A * x ≈ b` or `f(x) ≈ b` where `f`
encodes a linear map, i.e. a solution `x` that minimizes `norm(b - f(x))`.
Return the approximate solution `x` and a `ConvergenceInfo` structure.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. Since both the action of the linear map and its adjoint are required in
order to solve the least squares problem, `f` can either be a tuple of two callable objects
(each accepting a single argument), representing the linear map and its adjoint respectively, 
or, `f` can be a single callable object that accepts two input arguments, where the second
argument is a flag of type `Val{true}` or `Val{false}` that indicates whether the adjoint or
the normal action of the linear map needs to be computed. The latter form still combines
well with the `do` block syntax of Julia, as in

```julia
x, info = lssolve(b; kwargs...) do x, flag
    if flag === Val(true)
        # y = compute action of adjoint map on x
    else
        # y = compute action of linear map on x
    end
    return y
end
```

If the linear map `A` or `f` has a nontrivial nullspace, so different minimisers exist, the
solution being returned is such that `norm(x)` is minimal. Alternatively, the problem can
be providing a nonzero value for the optional argument `λ`, representing a scalar so 
that the minimisation problem `norm(b - A * x)^2 + λ * norm(x)^2` is solved instead.

!!! info "Starting guess"
    Note that `lssolve` does not allow to specify an starting guess `x₀` for the solution. The
    starting guess is always assumed to be the zero vector in the domain of the linear map, which
    is found by applying the adjoint action of the linear map to `b` and applying `zerovector`
    to the result. Given a good initial guess `x₀`, the user can call `lssolve` with a modified
    right hand side `b - f(x₀)` and add `x₀` to the solution returned by `lssolve`. The
    resulting vector `x` is a least squares solution to the original problem, but such that 
    `norm(x - x₀)` is minimal or `norm(b - A * x)^2 + λ * norm(x-x₀)^2` is minimised instead

### Return values:

The return value is always of the form `x, info = lssolve(...)` with

  - `x`: the least squares solution to the problem, as defined above 

  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: takes value 0 or 1 depending on whether the solution was
        converged up to the requested tolerance
      + `info.residual`: residual `b - A*x` of the approximate solution `x`
      + `info.normres::Real`: norm of the residual of the normal equations,
        i.e. the quantity `norm(A'*(b - A*x) - λ^2 * x)` that needs to be smaller
        than the requested tolerance `tol` in order to have a converged solution
      + `info.numops::Int`: total number of times that the linear map was applied, i.e. the
        number of times that `f` was called, or a vector was multiplied with `A` or `A'`
      + `info.numiter::Int`: total number of iterations of the algorithm

!!! warning "Check for convergence"

    No warning is printed if no converged solution was found, so always check if
    `info.converged == 1`.

### Keyword arguments:

Keyword arguments are given by:

  - `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
  - `atol::Real`: the requested accuracy, i.e. absolute tolerance, on the norm of the
    residual.
  - `rtol::Real`: the requested accuracy on the norm of the residual, relative to the norm
    of the right hand side `b`.
  - `tol::Real`: the requested accuracy on the norm of the residual that is actually used by
    the algorithm; it defaults to `max(atol, rtol*norm(b))`. So either use `atol` and `rtol`
    or directly use `tol` (in which case the value of `atol` and `rtol` will be ignored).
  - `maxiter::Integer`: the number of iterations of the algorithm. Every iteration involves
    one application of the linear map and one application of the adjoint of the linear map.

The default values are given by `atol = KrylovDefaults.tol`, `rtol = KrylovDefaults.tol`,
`tol = max(atol, rtol*norm(b))`, `maxiter = KrylovDefaults.maxiter`;
see [`KrylovDefaults`](@ref) for details.

### Algorithms

The final (expert) method, without default values and keyword arguments, is the one that is
finally called, and can also be used directly. Here, one specifies the algorithm explicitly.
Currently, only [`LSMR`](@ref) is available and thus selected.
"""
function lssolve end

function lssolve(f, b, λ=0;
                 maxiter=KrylovDefaults.maxiter,
                 rtol::Real=KrylovDefaults.tol,
                 atol::Real=KrylovDefaults.tol,
                 tol::Real=max(atol, rtol * norm(b)),
                 verbosity::Int=0)
    alg = LSMR(; maxiter, tol, verbosity)
    return lssolve(f, b, alg, λ)
end
