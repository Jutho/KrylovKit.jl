"""
    eigsolve(A::AbstractMatrix, [howmany = 1, which = :LM, T = eltype(A)]; kwargs...)
    eigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    eigsolve(f, x₀, [howmany = 1, which = :LM]; kwargs...)
    eigsolve(f, x₀, howmany, which, algorithm)

Compute at least `howmany` eigenvalues from the linear map encoded in the matrix `A` or by
the function `f`. Return eigenvalues, eigenvectors and a `ConvergenceInfo` structure.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. If an `AbstractMatrix` is used, a starting vector `x₀` does not need to be
provided, it is then chosen as `rand(T, size(A,1))`. If the linear map is encoded more
generally as a a callable function or method, the best approach is to provide an explicit
starting guess `x₀`. Note that `x₀` does not need to be of type `AbstractVector`, any type
that behaves as a vector and supports the required methods (see KrylovKit docs) is
accepted. If instead of `x₀` an integer `n` is specified, it is assumed that `x₀` is a
regular vector and it is initialized to `rand(T,n)`, where the default value of `T` is
`Float64`, unless specified differently.

The next arguments are optional, but should typically be specified. `howmany` specifies how
many eigenvalues should be computed; `which` specifies which eigenvalues should be
targetted. Valid specifications of `which` are given by
*   `:LM`: eigenvalues of largest magnitude
*   `:LR`: eigenvalues with largest (most positive) real part
*   `:SR`: eigenvalues with smallest (most negative) real part
*   `:LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
*   `:SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
*   [`EigSorter(f; rev = false)`](@ref): eigenvalues `λ` that appear first (or last if
    `rev == true`) when sorted by `f(λ)`
!!! note "Note about selecting `which` eigenvalues"
    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. All of they valid `Symbol`s for `which` have this
    property, but could also be specified usign `EigSorter`, e.g. `:LM` is equivalent to
    `Eigsorter(abs; rev = true)`. Note that smallest magnitude sorting is obtained using
    e.g. `EigSorter(abs; rev = false)`, but since no (shift-and)-invert is used, this will
    only be successfull if you somehow know that eigenvalues close to zero are also close
    to the periphery of the spectrum.

The argument `T` acts as a hint in which `Number` type the computation should be performed,
but is not restrictive. If the linear map automatically produces complex values, complex
arithmetic will be used even though `T<:Real` was specified. However, if the linear map and
initial guess are real, approximate eigenvalues will be searched for using a partial Schur
factorization, which implies that complex conjugate eigenvalues come in pairs and cannot
be split. It is then illegal to choose `which` in a way that would treat `λ` and `conj(λ)`
differently, i.e. `:LI` and `:SI` are invalid, as well as any `EigSorter` that would lead
to `by(λ) != by(conj(λ))`.

### Return values:
The return value is always of the form `vals, vecs, info = eigsolve(...)` with
*   `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could
    be longer if more eigenvalues were converged at the same cost. Eigenvalues will be real
    if [`Lanczos`](@ref) was used and complex if [`Arnoldi`](@ref) was used (see below).
*   `vecs`: a `Vector` of corresponding eigenvectors, of the same length as `vals`. Note
    that eigenvectors are not returned as a matrix, as the linear map could act on any
    custom Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`, up to a possibly
    different `eltype`. In particular  for a general matrix (i.e. with `Arnoldi`) the
    eigenvectors are generally complex and are therefore always returned in a complex
    number format. When the linear map is a simple `AbstractMatrix`, `vecs` will be
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
*   `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
*   `tol::Real`: the requested accuracy (corresponding to the 2-norm of the residual for
    Schur vectors, not the eigenvectors). If you work in e.g. single precision (`Float32`),
    you should definitely change the default value.
*   `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed. Note that the dimension of the vector space is not known or checked, e.g.
    `x₀` should not necessarily support the `Base.length` function. If you know the actual
    problem dimension is smaller than the default value, it is useful to reduce the value of
    `krylovdim`, though in principle this should be detected.
*   `maxiter::Integer`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
*   `orth::Orthogonalizer`: the orthogonalization method to be used, see
    [`Orthogonalizer`](@ref)
*   `issymmetric::Bool`: if the linear map is symmetric, only meaningful if `T<:Real`
*   `ishermitian::Bool`: if the linear map is hermitian
The default values are given by `tol = KrylovDefaults.tol`, `krylovdim =
KrylovDefaults.krylovdim`, `maxiter = KrylovDefaults.maxiter`,
`orth = KrylovDefaults.orth`; see [`KrylovDefaults`](@ref) for details.

The default value for the last two parameters depends on the method. If an `AbstractMatrix`
is used, `issymmetric` and `ishermitian` are checked for that matrix, ortherwise the default
values are `issymmetric = false` and `ishermitian = T <: Real && issymmetric`. When values
for the keyword arguments are provided, no checks will be performed even in the matrix case.

### Algorithm
The last method, without default values and keyword arguments, is the one that is finally
called, and can also be used directly. Here, one specifies the algorithm explicitly as
either [`Lanczos`](@ref), for real symmetric or complex hermitian problems, or
[`Arnoldi`](@ref), for general problems.
Note that these names refer to the process for building the Krylov subspace, but the actual
algorithm is an implementation of the Krylov-Schur algorithm, which can dynamically shrink
and grow the Krylov subspace, i.e. the restarts are so-called thick restarts where a part
of the current Krylov subspace is kept.

!!! note "Note about convergence"
    In case of a general problem, where the `Arnoldi` method is used, convergence of an
    eigenvalue is not based on the norm of the residual `norm(f(vecs[i]) -
    vals[i]*vecs[i])` for the eigenvector but rather on the norm of the residual for the
    corresponding Schur vectors.

    See also [`schursolve`](@ref) if you want to use the partial Schur decomposition
    directly, or if you are not interested in computing the eigenvectors, and want to work
    in real arithmetic all the way true (if the linear map and starting guess are real).
"""
function eigsolve end


"""
    EigSorter(by; rev = false)

A simple `struct` to be used in combination with [`eigsolve`](@ref) or [`schursolve`](@ref)
to indicate which eigenvalues need to be targetted, namely those that appear first when
sorted by `by` and possibly in reverse order if `rev == true`.
"""
struct EigSorter{F}
    by::F
    rev::Bool
end
EigSorter(f::F; rev = false) where F = EigSorter{F}(f, rev)

Base.@deprecate  ClosestTo(λ) EigSorter(z->abs(z-λ))

const Selector = Union{Symbol, EigSorter}

eigsolve(A::AbstractMatrix, howmany::Int = 1, which::Selector = :LM, T::Type = eltype(A);
    kwargs...) = eigsolve(A, rand(T, size(A,1)), howmany, which; kwargs...)

eigsolve(f, n::Int, howmany::Int = 1, which::Selector = :LM, T::Type = Float64; kwargs...) =
    eigsolve(f, rand(T, n), howmany, which; kwargs...)
function eigsolve(f, x₀, howmany::Int = 1, which::Selector = :LM; kwargs...)
    alg = eigselector(f, eltype(x₀); kwargs...)
    checkwhich(which) || error("Unknown eigenvalue selector: which = $which")
    if alg isa Lanczos
        if which == :LI || which == :SI
            error("Eigenvalue selector which = $which invalid: real eigenvalues expected with Lanczos algorithm")
        end
    elseif eltype(x₀) <: Real
        if which == :LI || which == :SI ||
            (which isa EigSorter && which.by(+im) != which.by(-im))

            error("Eigenvalue selector which = $which invalid because it does not treat
            `λ` and `conj(λ)` equally: work in complex arithmetic by providing a complex starting vector `x₀`")
        end
    end
    eigsolve(f, x₀, howmany, which, alg)
end

function eigselector(f, T::Type; issymmetric::Bool = false,
                                    ishermitian::Bool = T<:Real && issymmetric,
                                    krylovdim::Int = KrylovDefaults.krylovdim,
                                    maxiter::Int = KrylovDefaults.maxiter,
                                    tol::Real = KrylovDefaults.tol,
                                    orth::Orthogonalizer = KrylovDefaults.orth,
                                    verbosity::Int = 0)
    if (T<:Real && issymmetric) || ishermitian
        return Lanczos(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
        verbosity = verbosity)
    else
        return Arnoldi(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
        verbosity = verbosity)
    end
end
function eigselector(A::AbstractMatrix, T::Type;
                        issymmetric::Bool = eltype(A) <:Real && T <: Real &&
                                                LinearAlgebra.issymmetric(A),
                        ishermitian::Bool = issymmetric || LinearAlgebra.ishermitian(A),
                        krylovdim::Int = KrylovDefaults.krylovdim,
                        maxiter::Int = KrylovDefaults.maxiter,
                        tol::Real = KrylovDefaults.tol,
                        orth::Orthogonalizer = KrylovDefaults.orth,
                        verbosity::Int = 0)
    if (T<:Real && issymmetric) || ishermitian
        return Lanczos(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
        verbosity = verbosity)
    else
        return Arnoldi(krylovdim = krylovdim, maxiter = maxiter, tol = tol, orth = orth,
        verbosity= verbosity)
    end
end

checkwhich(::EigSorter) = true
# checkwhich(::ClosestTo) = true
checkwhich(s::Symbol) = s in (:LM, :LR, :SR, :LI, :SI)

eigsort(s::EigSorter) = s.by, s.rev
# function eigsort(which::ClosestTo)
#     by = x->abs(x-which.λ)
#     rev = false
#     return by, rev
# end
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
        error("invalid specification of which eigenvalues to target: which = $which")
    end
    return by, rev
end
