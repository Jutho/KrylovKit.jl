"""
    linsolve(A::AbstractMatrix, b::AbstractVector, T::Type = promote_type(eltype(A), eltype(b)); kwargs...)    eigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    linsolve(f, b, T::Type = eltype(b); kwargs...)
    linsolve(f, b, x₀; kwargs...)
    linsolve(f, b, x₀, algorithm)

Compute a solution `x` to the linear system `A*x = b` or `f(x) = b`, possibly using a starting
guess `x₀`. Return the approximate solution `x` and a `ConvergenceInfo` structure.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or callable
object. If no initial guess is specified, it is chosen as `fill!(similar(b, T), 0)`. The argument
`T` acts as a hint in which `Number` type the computation should be performed, but is not restrictive.
If the linear map automatically produces complex values, complex arithmetic will be used even
though `T<:Real` was specified.

### Return values:
The return value is always of the form `x, info = linsolve(...)` with
*   `x`: the approximate solution to the problem, similar type as the right hand side `b` but
    possibly with a different `eltype`
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: takes value 0 or 1 depending on whether the solution was converged
        up to the requested tolerance
    -   `info.residual`: residual `b - f(x)` of the approximate solution `x`
    -   `info.normres::Real`: norm of the residual, i.e. `norm(info.residual)`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged == 1`.

### Keyword arguments:
Keyword arguments and their default values are given by:
  * `krylovdim = 30`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should not
    necessarily support the `Base.length` function. If you know the actual problem dimension
    is smaller than the default value, it is useful to reduce the value of `krylovdim`, though
    in principle this should be detected.
  * `tol = 1e-12`: the requested accuracy (corresponding to the 2-norm of the residual for
    Schur vectors, not the eigenvectors). If you work in e.g. single precision (`Float32`),
    you should definitely change the default value.
  * `maxiter = 100`: the number of times the Krylov subspace can be rebuilt; see below for
    further details on the algorithms.
  * `issymmetric`: if the linear map is symmetric, only meaningful if `T<:Real`
  * `ishermitian`: if the linear map is hermitian
The default value for the last two depends on the method. If an `AbstractMatrix` is used,
`issymmetric` and `ishermitian` are checked for that matrix, ortherwise the default values are
`issymmetric = false` and `ishermitian = T <: Real && issymmetric`.
!!! warning "Use absolute tolerance"
    The keyword argument `tol` represents the actual tolerance, i.e. the solution `x` is deemed
    converged if `norm(b - f(x)) < tol`. If you want to use some relative tolerance `rtol`,
    use `tol = rtol*norm(b)` as keyword argument.

### Algorithm
The last method, without default values and keyword arguments, is the one that is finally called,
and can also be used directly. Here, one specifies the algorithm explicitly. Currently, only
[`GMRES`](@ref) is implemented. Note that we do not use the standard `GMRES` terminology, where
every new application of the linear map (matrix vector multiplications) counts as a new iteration,
and there is a `restart` parameter that indicates the largest Krylov subspace that is being built.
Indeed, the keyword argument `krylovdim` corresponds to the traditional `restart` parameter,
whereas `maxiter` counts the number of outer iterations, i.e. the number of times this Krylov
subspace is built. So the maximal number of operations of the linear map is roughly `maxiter * krylovdim`.
"""
function linsolve end

function linsolve(A::AbstractMatrix, b::AbstractVector, T::Type = promote_type(eltype(A), eltype(b)); kwargs...)
    linsolve(A, b, fill!(similar(b, T), 0); kwargs...)
end
function linsolve(f, b, T::Type = eltype(b); kwargs...)
    linsolve(A, b, fill!(similar(b, T), 0); kwargs...)
end
function linsolve(f, b, x₀; kwargs...)
    alg = linselector(f, eltype(x₀); kwargs...)
    linsolve(A, b, x₀, alg)
end

function linselector(f, T::Type;
    issymmetric = false, ishermitian = T<:Real && issymmetric, isposdef = false,
    krylovdim::Int = KrylovDefaults.krylovdim, maxiter::Int = KrylovDefaults.maxiter, tol::Real = KrylovDefaults.tol)
    if (T<:Real && issymmetric) || ishermitian
        # TODO
        # if isposdef
        #     return CG(krylovdim*maxiter, tol=tol)
        # else
        #     return MINRES(krylovdim*maxiter, tol=tol)
        # end
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol=tol)
    else
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol=tol)
    end
end
function linselector(A::AbstractMatrix, T::Type;
    issymmetric = T <: Real ? issymmetric(A) : false,
    ishermitian = T <: Complex ? ishermitian(A) : false,
    isposdef = issymmetric || ishermitian ? isposdef(A) : false,
    krylovdim::Int = KrylovDefaults.krylovdim, maxiter::Int = KrylovDefaults.maxiter, tol::Real = KrylovDefaults.tol,
    a0::Number = 0, a1::Number = 1)
    if (T<:Real && issymmetric) || ishermitian
        # TODO
        if isposdef
            return CG(krylovdim*maxiter, tol=tol, a0, a1)
        # else
        #     return MINRES(krylovdim*maxiter, tol=tol)
        end
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol=tol, a0, a1)
    else
        return GMRES(krylovdim = krylovdim, maxiter = maxiter, tol=tol, a0, a1)
    end
end
