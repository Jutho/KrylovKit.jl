"""
    geneigsolve(A::AbstractMatrix, B::AbstractMatrix, [howmany = 1, which = :LM, T = promote_type(eltype(A), eltype(B))]; kwargs...)
    geneigsolve(A, B, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    geneigsolve(A, B, x₀, [howmany = 1, which = :LM]; kwargs...)
    geneigsolve(A, B, x₀, howmany, which, algorithm)

Compute `howmany` generalized eigenvalues ``λ`` and generalized eigenvectors ``x`` of the
form ``(A - λB)x = 0``, where `A` and `B` are either instances of `AbstractMatrix`, or some
function that implements the matrix vector product. Return eigenvalues, eigenvectors and a `ConvergenceInfo` structure.

### Arguments:
The linear maps `A` and `B` can be an `AbstractMatrix` (dense or sparse) or a general
function or callable object. If an `AbstractMatrix` is used for either `A` or `B`, a
starting vector `x₀` does not need to be provided, it is then chosen as
`rand(T, size(A,1))` if `A` is an `AbstractMatrix` (or similarly if only `B` is an
`AbstractMatrix`). Here `T = promote_type(eltype(A), eltype(B))` if both `A` and `B` are
instances of `AbstractMatrix`, or just the `eltype` of whichever is an `AbstractMatrix`. If
both `A` and `B` are encoded more generally as a callable function or method, the best
approach is to provide an explicit starting guess `x₀`. Note that `x₀` does not need to be
of type `AbstractVector`, any type that behaves as a vector and supports the required
methods (see KrylovKit docs) is accepted. If instead of `x₀` an integer `n` is specified,
it is assumed that `x₀` is a regular vector and it is initialized to `rand(T,n)`, where the
default value of `T` is `Float64`, unless specified differently.

The next arguments are optional, but should typically be specified. `howmany` specifies how many eigenvalues should be computed; `which` specifies which eigenvalues should be targetted. Valid specifications of `which` are given by
*   `LR`: eigenvalues with largest (most positive) real part
*   `SR`: eigenvalues with smallest (most negative) real part
!!! note "Note about selecting `which` eigenvalues"
    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. Even with `ClosestTo`, no shift and invert is performed.
    This is useful if, e.g., you know the spectrum to be within the unit circle in the complex
    plane, and want to target the eigenvalues closest to the value `λ = 1`.

The argument `T` acts as a hint in which `Number` type the computation should be performed, but
is not restrictive. If the linear map automatically produces complex values, complex arithmetic
will be used even though `T<:Real` was specified.

### Return values:
The return value is always of the form `vals, vecs, info = eigsolve(...)` with
*   `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could
    be longer if more eigenvalues were converged at the same cost.
*   `vecs`: a `Vector` of corresponding eigenvectors, of the same length as `vals`.
    Note that eigenvectors are not returned as a matrix, as the linear map could act on any
    custom Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`, up to a possibly
    different `eltype`. When the linear map is a simple `AbstractMatrix`, `vecs` will be
    `Vector{Vector{<:Number}}`.
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: indicates how many eigenvalues and eigenvectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
    -   `info.residual::Vector`: a list of the same length as `vals` containing the
        residuals `info.residual[i] = f(vecs[i]) - vals[i] * vecs[i]`
    -   `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Keyword arguments:
Keyword arguments and their default values are given by:
*   `tol::Real`: the requested accuracy (corresponding to the 2-norm of the residual for
    Schur vectors, not the eigenvectors). If you work in e.g. single precision (`Float32`),
    you should definitely change the default value.
*   `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed. Note that the dimension of the vector space is not known or checked, e.g.
    `x₀` should not necessarily support the `Base.length` function. If you know the actual
    problem dimension is smaller than the default value, it is useful to reduce the value
    of `krylovdim`, though in principle this should be detected.
*   `maxiter::Integer`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
*   `orth::Orthogonalizer`: the orthogonalization method to be used, see
    [`Orthogonalizer`](@ref)
*   `issymmetric::Bool`: if both linear maps `A` and `B` are symmetric, only meaningful if
    `T<:Real`
*   `ishermitian::Bool`: if both linear maps `A` and `B` are hermitian
*   `isposdef::Bool`: if the linear map `B` is positive definites
The default values are given by `tol = KrylovDefaults.tol`, `krylovdim =
KrylovDefaults.krylovdim`, `maxiter = KrylovDefaults.maxiter`, `orth =
KrylovDefaults.orth`; see [`KrylovDefaults`](@ref) for details.

The default value for the last three parameters depends on the method. If an
`AbstractMatrix` is used, `issymmetric`, `ishermitian` and `isposdef` are checked for that
matrix, ortherwise the default values are `issymmetric = false` and `ishermitian = T <:
Real && issymmetric`. When values are provided, no checks will be performed even in the
matrix case.

### Algorithm
The last method, without default values and keyword arguments, is the one that is finally called,
and can also be used directly. Here, one specifies the algorithm explicitly as either [`Lanczos`](@ref),
for real symmetric or complex hermitian problems, or [`Arnoldi`](@ref), for general problems.
Note that these names refer to the process for building the Krylov subspace, but the actual
algorithm is an implementation of the Krylov-Schur algorithm, which can dynamically shrink and
grow the Krylov subspace, i.e. the restarts are so-called thick restarts where a part of the
current Krylov subspace is kept.

"""
function geneigsolve end

function geneigsolve(A::AbstractMatrix, B::AbstractMatrix, howmany::Int = 1,
        which::Selector = :LM, T = promote_type(eltype(A), eltype(B)); kwargs...)
    if !(size(A,1) == size(A,2) == size(B,1) == size(B,2))
        throw(DimensionMismatch("Matrices `A` and `B` should be square and have matching size"))
    end
    geneigsolve(A, B, rand(T, size(A,1)), howmany::Int, which; kwargs...)
end
geneigsolve(A, B::AbstractMatrix, howmany::Int = 1, which::Selector = :LM, T = eltype(B);
    kwargs...) = geneigsolve(A, B, rand(T, size(B,1)), howmany, which; kwargs...)
geneigsolve(A::AbstractMatrix, B, howmany::Int = 1, which::Selector = :LM, T = eltype(A);
    kwargs...) = geneigsolve(A, B, rand(T, size(A,1)), howmany, which; kwargs...)

geneigsolve(A, B, n::Int, howmany::Int = 1, which::Selector = :LM, T = Float64; kwargs...) =
    geneigsolve(A, B, rand(T, n), howmany, which; kwargs...)

function geneigsolve(A, B, x₀, howmany::Int = 1, which::Selector = :LM; kwargs...)
    alg = geneigselector(A, B, eltype(x₀); kwargs...)
    if alg isa GolubYe && (which == :LI || which == :SI)
        error("Eigenvalue selector which = $which invalid: real eigenvalues expected with Lanczos algorithm")
    end
    geneigsolve(A, B, x₀, howmany, which, alg)
end

function geneigselector(A::AbstractMatrix, B::AbstractMatrix, T::Type;
                        issymmetric = eltype(A) <: Real && eltype(B) <:Real && T <: Real &&
                                        all(LinearAlgebra.issymmetric, (A,B)),
                        ishermitian = issymmetric || all(LinearAlgebra.ishermitian, (A,B)),
                        isposdef = ishermitian && LinearAlgebra.isposdef(B), kwargs...)
    if (issymmetric || ishermitian) && isposdef
        return GolubYe(; kwargs...)
    else
        throw(ArgumentError("Only symmetric or hermitian generalized eigenvalue problems with positive definite `B` matrix are currently supported."))
    end
end
function geneigselector(A, B, T::Type; issymmetric = false, ishermitian = issymmetric,
                            isposdef = false, kwargs...)
    if (issymmetric || ishermitian) && isposdef
        return GolubYe(; kwargs...)
    else
        throw(ArgumentError("Only symmetric or hermitian generalized eigenvalue problems with positive definite `B` matrix are currently supported."))
    end
end
