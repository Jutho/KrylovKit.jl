"""
    eigsolve(A::AbstractMatrix, [x₀, howmany = 1, which = :LM, T = eltype(A)]; kwargs...)
    eigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    eigsolve(f, x₀, [howmany = 1, which = :LM]; kwargs...)
    # expert version:
    eigsolve(f, x₀, howmany, which, algorithm; alg_rrule=...)

Compute at least `howmany` eigenvalues from the linear map encoded in the matrix `A` or by
the function `f`. Return eigenvalues, eigenvectors and a `ConvergenceInfo` structure.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. If an `AbstractMatrix` is used, a starting vector `x₀` does not need to be
provided, it is then chosen as `rand(T, size(A, 1))`. If the linear map is encoded more
generally as a a callable function or method, the best approach is to provide an explicit
starting guess `x₀`. Note that `x₀` does not need to be of type `AbstractVector`; any type
that behaves as a vector and supports the required methods (see KrylovKit docs) is accepted.
If instead of `x₀` an integer `n` is specified, it is assumed that `x₀` is a regular vector
and it is initialized to `rand(T, n)`, where the default value of `T` is `Float64`, unless
specified differently.

The next arguments are optional, but should typically be specified. `howmany` specifies how
many eigenvalues should be computed; `which` specifies which eigenvalues should be
targeted. Valid specifications of `which` are given by

  - `:LM`: eigenvalues of largest magnitude
  - `:LR`: eigenvalues with largest (most positive) real part
  - `:SR`: eigenvalues with smallest (most negative) real part
  - `:LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
  - `:SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
  - [`EigSorter(f; rev = false)`](@ref): eigenvalues `λ` that appear first (or last if
    `rev == true`) when sorted by `f(λ)`

!!! note "Note about selecting `which` eigenvalues"

    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. All of the valid `Symbol`s for `which` have this
    property, but could also be specified using `EigSorter`, e.g. `:LM` is equivalent to
    `Eigsorter(abs; rev = true)`. Note that smallest magnitude sorting is obtained using
    e.g. `EigSorter(abs; rev = false)`, but since no (shift-and)-invert is used, this will
    only be successful if you somehow know that eigenvalues close to zero are also close
    to the periphery of the spectrum.

!!! warning "Degenerate eigenvalues"

    From a theoretical point of view, Krylov methods can at most find a single eigenvector
    associated with a targetted eigenvalue, even if the latter is degenerate. In the case of
    a degenerate eigenvalue, the specific eigenvector that is returned is determined by the
    starting vector `x₀`. For large problems, this turns out to be less of an issue in
    practice, as often a second linearly independent eigenvector is generated out of the
    numerical noise resulting from the orthogonalisation steps in the Lanczos or Arnoldi
    iteration. Nonetheless, it is important to take this into account and to try not to
    depend on this potentially fragile behaviour, especially for smaller problems.

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

  - `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could
    be longer if more eigenvalues were converged at the same cost. Eigenvalues will be real
    if [`Lanczos`](@ref) was used and complex if [`Arnoldi`](@ref) was used (see below).
  - `vecs`: a `Vector` of corresponding eigenvectors, of the same length as `vals`. Note
    that eigenvectors are not returned as a matrix, as the linear map could act on any
    custom Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`, up to a possibly
    different `eltype`. In particular  for a general matrix (i.e. with `Arnoldi`) the
    eigenvectors are generally complex and are therefore always returned in a complex
    number format. When the linear map is a simple `AbstractMatrix`, `vecs` will be
    `Vector{Vector{<:Number}}`.
  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: indicates how many eigenvalues and eigenvectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
      + `info.residual::Vector`: a list of the same length as `vals` containing the
        residuals `info.residual[i] = f(vecs[i]) - vals[i] * vecs[i]`
      + `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
      + `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

!!! warning "Check for convergence"

    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Keyword arguments:

Keyword arguments and their default values are given by:

  - `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
  - `tol::Real`: the requested accuracy (corresponding to the 2-norm of the residual for
    Schur vectors, not the eigenvectors). If you work in e.g. single precision (`Float32`),
    you should definitely change the default value.
  - `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed. Note that the dimension of the vector space is not known or checked, e.g.
    `x₀` should not necessarily support the `Base.length` function. If you know the actual
    problem dimension is smaller than the default value, it is useful to reduce the value of
    `krylovdim`, though in principle this should be detected.
  - `maxiter::Integer`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
  - `orth::Orthogonalizer`: the orthogonalization method to be used, see
    [`Orthogonalizer`](@ref)
  - `issymmetric::Bool`: if the linear map is symmetric, only meaningful if `T<:Real`
  - `ishermitian::Bool`: if the linear map is hermitian
  - `eager::Bool = false`: if true, eagerly compute the eigenvalue or Schur decomposition
    after every expansion of the Krylov subspace to test for convergence, otherwise wait
    until the Krylov subspace has dimension `krylovdim`. This can result in a faster return,
    for example if the initial guess is very good, but also has some overhead, as many more
    dense Schur factorizations need to be computed.

The default values are given by `tol = KrylovDefaults.tol`,
`krylovdim = KrylovDefaults.krylovdim`, `maxiter = KrylovDefaults.maxiter`,
`orth = KrylovDefaults.orth`; see [`KrylovDefaults`](@ref) for details.

The default value for the last two parameters depends on the method. If an `AbstractMatrix`
is used, `issymmetric` and `ishermitian` are checked for that matrix, otherwise the default
values are `issymmetric = false` and `ishermitian = T <: Real && issymmetric`. When values
for the keyword arguments are provided, no checks will be performed even in the matrix case.

The final keyword argument `alg_rrule` is relevant only when `eigsolve` is used in a setting
where reverse-mode automatic differentation will be used. A custom `ChainRulesCore.rrule` is
defined for `eigsolve`, which can be evaluated using different algorithms that can be specified
via `alg_rrule`. A suitable default is chosen, so this keyword argument should only be used
when this default choice is failing or not performing efficiently. Check the documentation for
more information on the possible values for `alg_rrule` and their implications on the algorithm
being used.

### Algorithm

The final (expert) method, without default values and keyword arguments, is the one that is
finally called, and can also be used directly. Here, one specifies the algorithm explicitly
as either [`Lanczos`](@ref), for real symmetric or complex hermitian problems, or
[`Arnoldi`](@ref), for general problems. Note that these names refer to the process for
building the Krylov subspace, but the actual algorithm is an implementation of the
Krylov-Schur algorithm, which can dynamically shrink and grow the Krylov subspace, i.e. the
restarts are so-called thick restarts where a part of the current Krylov subspace is kept.

!!! note "Note about convergence"

    In case of a general problem, where the `Arnoldi` method is used, convergence of an
    eigenvalue is not based on the norm of the residual `norm(f(vecs[i]) - vals[i]*vecs[i])`
    for the eigenvector but rather on the norm of the residual for the corresponding Schur
    vectors.

    See also [`schursolve`](@ref) if you want to use the partial Schur decomposition
    directly, or if you are not interested in computing the eigenvectors, and want to work
    in real arithmetic all the way true (if the linear map and starting guess are real).
    If you have knowledge that all requested eigenvalues of a real problem will be real,
    and thus also their associated eigenvectors, you can also use [`realeigsolve`](@ref).
"""
function eigsolve end

"""
    EigSorter(by; rev = false)

A simple `struct` to be used in combination with [`eigsolve`](@ref) or [`schursolve`](@ref)
to indicate which eigenvalues need to be targeted, namely those that appear first when
sorted by `by` and possibly in reverse order if `rev == true`.
"""
struct EigSorter{F}
    by::F
    rev::Bool
end
EigSorter(f::F; rev=false) where {F} = EigSorter{F}(f, rev)

const Selector = Union{Symbol,EigSorter}

function eigsolve(A::AbstractMatrix,
                  howmany::Int=1,
                  which::Selector=:LM,
                  T::Type=eltype(A);
                  kwargs...)
    x₀ = Random.rand!(similar(A, T, size(A, 1)))
    return eigsolve(A, x₀, howmany, which; kwargs...)
end

function eigsolve(f, n::Int, howmany::Int=1, which::Selector=:LM, T::Type=Float64;
                  kwargs...)
    return eigsolve(f, rand(T, n), howmany, which; kwargs...)
end
function eigsolve(f, x₀, howmany::Int=1, which::Selector=:LM; kwargs...)
    Tx = typeof(x₀)
    Tfx = Core.Compiler.return_type(apply, Tuple{typeof(f),Tx})
    T = Core.Compiler.return_type(dot, Tuple{Tx,Tfx})
    alg = eigselector(f, T; kwargs...)
    checkwhich(which) || error("Unknown eigenvalue selector: which = $which")
    if alg isa Lanczos
        if which == :LI || which == :SI
            error("Eigenvalue selector which = $which invalid: real eigenvalues expected with Lanczos algorithm")
        end
    elseif T <: Real
        by, rev = eigsort(which)
        if by(+im) != by(-im)
            error("Eigenvalue selector which = $which invalid because it does not treat
            `λ` and `conj(λ)` equally: work in complex arithmetic by providing a complex starting vector `x₀`")
        end
    end
    if haskey(kwargs, :alg_rrule)
        alg_rrule = kwargs[:alg_rrule]
    else
        alg_rrule = Arnoldi(; tol=alg.tol,
                            krylovdim=alg.krylovdim,
                            maxiter=alg.maxiter,
                            eager=alg.eager,
                            orth=alg.orth)
    end
    return eigsolve(f, x₀, howmany, which, alg; alg_rrule=alg_rrule)
end

function eigselector(f,
                     T::Type;
                     issymmetric::Bool=false,
                     ishermitian::Bool=issymmetric && (T <: Real),
                     krylovdim::Int=KrylovDefaults.krylovdim,
                     maxiter::Int=KrylovDefaults.maxiter,
                     tol::Real=KrylovDefaults.tol,
                     orth::Orthogonalizer=KrylovDefaults.orth,
                     eager::Bool=false,
                     verbosity::Int=0,
                     alg_rrule=nothing)
    if (T <: Real && issymmetric) || ishermitian
        return Lanczos(; krylovdim=krylovdim,
                       maxiter=maxiter,
                       tol=tol,
                       orth=orth,
                       eager=eager,
                       verbosity=verbosity)
    else
        return Arnoldi(; krylovdim=krylovdim,
                       maxiter=maxiter,
                       tol=tol,
                       orth=orth,
                       eager=eager,
                       verbosity=verbosity)
    end
end
function eigselector(A::AbstractMatrix,
                     T::Type;
                     issymmetric::Bool=T <: Real && LinearAlgebra.issymmetric(A),
                     ishermitian::Bool=issymmetric || LinearAlgebra.ishermitian(A),
                     krylovdim::Int=KrylovDefaults.krylovdim,
                     maxiter::Int=KrylovDefaults.maxiter,
                     tol::Real=KrylovDefaults.tol,
                     orth::Orthogonalizer=KrylovDefaults.orth,
                     eager::Bool=false,
                     verbosity::Int=0,
                     alg_rrule=nothing)
    if (T <: Real && issymmetric) || ishermitian
        return Lanczos(; krylovdim=krylovdim,
                       maxiter=maxiter,
                       tol=tol,
                       orth=orth,
                       eager=eager,
                       verbosity=verbosity)
    else
        return Arnoldi(; krylovdim=krylovdim,
                       maxiter=maxiter,
                       tol=tol,
                       orth=orth,
                       eager=eager,
                       verbosity=verbosity)
    end
end

checkwhich(::EigSorter) = true
checkwhich(s::Symbol) = s in (:LM, :LR, :SR, :LI, :SI)

eigsort(s::EigSorter) = s.by, s.rev
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
