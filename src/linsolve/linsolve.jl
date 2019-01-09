"""
    linsolve(A::AbstractMatrix, b::AbstractVector, [a₀::Number = 0, a₁::Number = 1,
                T::Type = promote_type(eltype(A), eltype(b), typeof(a₀), typeof(a₁))];
                kwargs...)
    linsolve(f, b, [a₀::Number = 0, a₁::Number = 1,
                T::Type = promote_type(eltype(b), typeof(a₀), typeof(a₁))]; kwargs...)
    linsolve(f, b, x₀, [a₀::Number = 0, a₁::Number = 1]; kwargs...)
    linsolve(f, b, x₀, algorithm, [a₀::Number = 0, a₁::Number = 1])

Compute a solution `x` to the linear system `(a₀ + a₁ * A)*x = b` or `a₀ * x + a₁ * f(x) =
b`, possibly using a starting guess `x₀`. Return the approximate solution `x` and a
`ConvergenceInfo` structure.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. If no initial guess is specified, it is chosen as `rmul!(similar(b, T),
false)` which generates a similar object to `b`, but with element type `T` and initialized
with zeros. The numbers `a₀` and `a₁` are optional arguments; they are applied implicitly,
i.e. they do not contribute the computation time of applying the linear map or to the number
of operations on vectors of type `x` and `b`.

Finally, the optional argument `T` acts as a hint in which `Number` type the computation
should be performed, but is not restrictive. If the linear map automatically produces
complex values, complex arithmetic will be used even though `T<:Real` was specified.

### Return values:
The return value is always of the form `x, info = linsolve(...)` with
*   `x`: the approximate solution to the problem, similar type as the right hand side `b`
    but possibly with a different `eltype`
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: takes value 0 or 1 depending on whether the solution was
        converged up to the requested tolerance
    -   `info.residual`: residual `b - f(x)` of the approximate solution `x`
    -   `info.normres::Real`: norm of the residual, i.e. `norm(info.residual)`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged == 1`.

### Keyword arguments:
Keyword arguments are given by:
*   `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
*   `atol::Real`: the requested accuracy, i.e. absolute tolerance, on the norm of the
    residual.
*   `rtol::Real`: the requested accuracy on the norm of the residual, relative to the norm
    of the right hand side `b`.
*   `tol::Real`: the requested accuracy on the norm of the residual which is actually used,
    but which defaults to `max(atol, rtol*norm(b))`. So either `atol` and `rtol` or directly
    use `tol` (in which case the value of `atol` and `rtol` will be ignored).
*   `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed.
*   `maxiter::Integer: the number of times the Krylov subspace can be rebuilt; see below for
    further details on the algorithms.
*   `orth::Orthogonalizer`: the orthogonalization method to be used, see
    [`Orthogonalizer`](@ref)
*   `issymmetric::Bool`: if the linear map is symmetric, only meaningful if `T<:Real`
*   `ishermitian::Bool`: if the linear map is hermitian
*   `isposdef::Bool`: if the linear map is positive definite
The default values are given by `atol = KrylovDefaults.tol`, `rtol = KrylovDefaults.tol`,
`tol = max(atol, rtol*norm(b))`, `krylovdim = KrylovDefaults.krylovdim`,
`maxiter = KrylovDefaults.maxiter`, `orth = KrylovDefaults.orth`;
see [`KrylovDefaults`](@ref) for details.

The default value for the last three parameters depends on the method. If an
`AbstractMatrix` is used, `issymmetric`, `ishermitian` and `isposdef` are checked for that
matrix, ortherwise the default values are `issymmetric = false`, `ishermitian = T <: Real
&& issymmetric` and `isposdef = false`.

### Algorithms
The last method, without default values and keyword arguments, is the one that is finally
called, and can also be used directly. Here, one specifies the algorithm explicitly.
Currently, only [`CG`](@ref) and [`GMRES`](@ref) are implemented, where `CG` is chosen if
`isposdef == true`. Note that in standard `GMRES` terminology, our parameter `krylovdim` is
referred to as the *restart* parameter, and our `maxiter` parameter counts the number of
outer iterations, i.e. restart cycles. In `CG`, the Krylov subspace is only implicit
because short recurrence relations are being used, and therefore no restarts are required.
Therefore, we pass `krylovdim*maxiter` as the maximal number of CG iterations that can be
used by the `CG` algorithm.
"""
function linsolve end

linsolve(A::AbstractMatrix, b::AbstractVector, a₀::Number = 0, a₁::Number = 1,
            T::Type = promote_type(eltype(A), eltype(b), typeof(a₀), typeof(a₁));
            kwargs...) = linsolve(A, b, rmul!(similar(b, T), false), a₀, a₁; kwargs...)

linsolve(f, b, a₀::Number = 0, a₁::Number = 1,
            T::Type = promote_type(eltype(b), typeof(a₀), typeof(a₁));
            kwargs...) = linsolve(f, b, rmul!(similar(b, T), false), a₀, a₁; kwargs...)

function linsolve(f, b, x₀, a₀::Number = 0, a₁::Number = 1; kwargs...)
    alg = linselector(f, b, promote_type(eltype(x₀), typeof(a₀), typeof(a₁)); kwargs...)
    linsolve(f, b, x₀, alg, a₀, a₁)
end

function linselector(f, b, T::Type; issymmetric::Bool = false,
                                    ishermitian::Bool = T<:Real && issymmetric,
                                    isposdef::Bool = false,
                                    krylovdim::Int = KrylovDefaults.krylovdim,
                                    maxiter::Int = KrylovDefaults.maxiter,
                                    rtol::Real = KrylovDefaults.tol,
                                    atol::Real = KrylovDefaults.tol,
                                    tol::Real = max(atol, rtol*norm(b)),
                                    orth = KrylovDefaults.orth,
                                    verbosity::Int = 0)
    if (T<:Real && issymmetric) || ishermitian
        if isposdef
            return CG(maxiter = krylovdim*maxiter, tol = tol, verbosity = verbosity)
        # else
        # TODO
        #     return MINRES(krylovdim*maxiter, tol=tol)
        end
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
                        verbosity = verbosity)
    else
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
                        verbosity = verbosity)
    end
end
function linselector(A::AbstractMatrix, b, T::Type;
                        issymmetric::Bool = T <: Real && LinearAlgebra.issymmetric(A),
                        ishermitian::Bool = issymmetric || LinearAlgebra.ishermitian(A),
                        isposdef::Bool = ishermitian ? LinearAlgebra.isposdef(A) : false,
                        krylovdim::Int = KrylovDefaults.krylovdim,
                        maxiter::Int = KrylovDefaults.maxiter,
                        rtol::Real = KrylovDefaults.tol,
                        atol::Real = KrylovDefaults.tol,
                        tol::Real = max(atol, rtol*norm(b)),
                        orth = KrylovDefaults.orth,
                        verbosity::Int = 0)
    if (T<:Real && issymmetric) || ishermitian
        if isposdef
            return CG(maxiter = krylovdim*maxiter, tol = tol, verbosity = verbosity)
        # else
        # TODO
        #     return MINRES(krylovdim*maxiter, tol=tol, verbosity = verbosity)
        end
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
                        verbosity = verbosity)
    else
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
                        verbosity = verbosity)
    end
end
