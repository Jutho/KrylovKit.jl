"""
    eigsolve(A::AbstractMatrix, [x₀, howmany = 1, which = :LM, T = eltype(A)]; kwargs...)
    eigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    eigsolve(f, x₀, [howmany = 1, which = :LM, T = eltype(x₀)]; kwargs...)
    eigsolve(f, x₀, howmany = 1, which = :LM, T = eltype(x₀); kwargs...)
    eigsolve(f, x₀, howmany, which, algorithm)

Compute `howmany` eigenvalues from the linear map encoded in the matrix `A` or by the
function `f`. Return eigenvalues, eigenvectors and a `ConvergenceInfo` structure.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or callable
object. If an `AbstractMatrix` is used, a starting vector `x₀` does not need to be provided,
it is then chosen as `rand(T, size(A,1))`. If the linear map is encoded more generally as a
a callable function or method, the best approach is to provide an explicit starting guess `x₀`.
Note that `x₀` does not need to be of type `AbstractVector`, any type that behaves as a vector
and supports the required methods (see KrylovKit docs) is accepted. If instead of `x₀` an integer
`n` is specified, it is assumed that `x₀` is a regular vector and it is initialized to `rand(T,n)`,
where the default value of `T` is `Float64`, unless specified differently.

The next arguments are optional, but should typically be specified. `howmany` specifies how many
eigenvalues should be computed; `which` specifies which eigenvalues should be targetted. Valid
specifications of `which` are
  * `LM`: eigenvalues of largest magnitude
  * `LR`: eigenvalues with largest (most positive) real part
  * `SR`: eigenvalues with smallest (most negative) real part
  * `LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
  * `SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
  * [`ClosestTo(λ)`](@ref): eigenvalues closest to some number `λ`
!!! note
    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. Even with `ClosestTo`, no shift and invert is performed.
    This is useful if, e.g., you know the spectrum to be within the unit circle in the complex
    plane, and want to target the eigenvalues closest to the value `λ = 1`.

The argument `T` acts as a hint in which `Number` type the computation should be performed, but
is not restrictive. If the linear map automatically produces complex values, complex arithmetic
will be used even though `T<:Real` was specified.

### Return values:
The return value is always of the form `vals, vecs, info = eigsolve(...)` with
  * `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could be
    longer if more eigenvalues were converged at the same cost. Eigenvalues will be real if
    [`Lanczos`](@ref) was used and complex if [`Arnoldi`](@ref) was used (see below).
  * `vecs`: a `Vector` of corresponding eigenvectors, of the same length as `vals`. Note that
    eigenvectors are not returned as a matrix, as the linear map could act on any custom Julia
    type with vector like behavior, i.e. the elements of the list `vecs` are objects that are
    typically similar to the starting guess `x₀`, up to a possibly different `eltype`.
    When the linear map is a simple `AbstractMatrix`, `vecs` will be `Vector{Vector{<:Number}}`.
  * `info`: an object of type [`ConvergenceInfo`], which has the following fields
      - `info.converged::Int`: indices how many eigenvalues and eigenvectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
      - `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the norm
        of the residual for every eigenvector, i.e. `info.normres[i] = norm(f(vecs[i]) - vals[i] * vecs[i])`
      - `info.residuals::Vector`: a list of the same length as `vals` containing the actual
        residuals `info.residuals[i] = f(vecs[i]) - vals[i] * vecs[i]`
      - `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
      - `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

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

### Algorithm
The last method, without default values and keyword arguments, is the one that is finally called,
and can also be used directly. Here, one specifies the algorithm explicitly as either [`Lanczos`](@ref),
for real symmetric or complex hermitian problems, or [`Arnoldi`](@ref), for general problems.
Note that these names refer to the process for building the Krylov subspace, but the actual
algorithm is an implementation of the Krylov-Schur algorithm, which can dynamically shrink and
grow the Krylov subspace, i.e. the restarts are so-called thick restarts where a part of the
current Krylov subspace is kept.

!!! note "Note about convergence"
    In case of a general problem, where the `Arnoldi` method is used, convergence of an eigenvalue
    is not based on the norm of the residual `norm(f(vecs[i]) - vals[i]*vecs[i])` for the eigenvectors
    but rather on the norm of the residual for the corresponding Schur vectors.

    See also [`schursolve`](@ref) if you want to use the partial Schur decomposition directly,
    or if you are not interested in computing the eigenvectors, and want to work in real arithetic
    all the way true.
"""
function eigsolve end


"""
    ClosestTo(λ)

A simple `struct` to be used in combination with [`eigsolve`](@ref) or [`schursolve`](@ref) to
indicate which eigenvalues need to be targetted, namely those closest to `λ`.
"""
struct ClosestTo{T}
    λ::T
end

const Selector = Union{ClosestTo, Symbol}

function eigsolve(A::AbstractMatrix, howmany::Int = 1, which::Selector = :LM, T::Type = eltype(A);
        issymmetric = issymmetric(A), ishermitian = ishermitian(A),
        krylovdim::Int = KrylovDefaults.krylovdim, maxiter::Int = KrylovDefaults.maxiter, tol::Real = KrylovDefaults.tol)
    eigsolve(x->(A*x), size(A,1), howmany, which, T; issymmetric = issymmetric, ishermitian = ishermitian, krylovdim = krylovdim, maxiter = maxiter, tol = tol)
end

function eigsolve(A::AbstractMatrix, x₀::VecOrMat, howmany::Int = 1, which::Selector = :LM, T::Type = promote_type(eltype(A), eltype(x₀));
        issymmetric = issymmetric(A), ishermitian = ishermitian(A),
        krylovdim::Int = KrylovDefaults.krylovdim, maxiter::Int = KrylovDefaults.maxiter, tol::Real = KrylovDefaults.tol)
    eigsolve(x->(A*x), x₀, howmany, which, T; issymmetric = issymmetric, ishermitian = ishermitian, krylovdim = krylovdim, maxiter = maxiter, tol = tol)
end

eigsolve(f, n::Int, howmany::Int = 1, which::Selector = :LM, T::Type = Float64; kwargs...) =
    eigsolve(f, rand(T, n), howmany, which; kwargs...)

function eigsolve(f, x₀, howmany::Int = 1, which::Selector = :LM, T::Type = eltype(x₀);
        issymmetric = false, ishermitian = T<:Real && issymmetric,
        krylovdim::Int = KrylovDefaults.krylovdim, maxiter::Int = KrylovDefaults.maxiter, tol::Real = KrylovDefaults.tol)
    x = eltype(x₀) == T ? x₀ : copyto!(similar(x₀, T), x₀)
    if T<:Real
        (which == :LI || which == :SI) && throw(ArgumentError("work in complex domain to find eigenvalues with largest or smallest imaginary part"))
    end
    if (T<:Real && issymmetric) || ishermitian
        return eigsolve(f, x, howmany, which, Lanczos(krylovdim = krylovdim, maxiter = maxiter, tol=tol))
    else
        return eigsolve(f, x, howmany, which, Arnoldi(krylovdim = krylovdim, maxiter = maxiter, tol=tol))
    end
end


function eigsort(which::ClosestTo)
    by = x->abs(x-which.λ)
    rev = false
    return by, rev
end
function eigsort(which::Symbol)
    if which == :LM
        by = abs
        rev = true
    elseif which == :LR
        by = real
        rev = true
    elseif which == :SR
        by = real
        rev = false
    elseif which == :LI
        by = imag
        rev = true
    elseif which == :SI
        by = imag
        rev = false
    else
        error("incorrect value of which: $which")
    end
    return by, rev
end
