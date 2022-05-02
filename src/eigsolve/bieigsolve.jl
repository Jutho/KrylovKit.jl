"""
    bieigsolve(A::AbstractMatrix, [x₀, y₀ = x₀, T = eltype(A)]; 
        howmany = 1, which = :LM, kwargs...)
    bieigsolve(f, n::Int, [T = Float64]; howmany = 1, which = :LM, kwargs...)
    bieigsolve(f, x₀, [y₀]; howmany = 1, which = :LM, kwargs...)
    # expert version:
    bieigsolve(f, x₀, y₀, algorithm; howmany = 1, which = :LM)

Compute at least `howmany` eigenvalues for the linear map encoded in the matrix `A` or by
the function `f`. Return eigenvalues, left and right eigenvectors and a `ConvergenceInfo`
structure.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. Since both the action of the linear map and its adjoint are required in
order to compute left and right eigenvectors, `f` can either be a tuple of two callable
objects (each accepting a single argument), representing the linear map and its adjoint
respectively, or, `f` can be a single callable object that accepts two input arguments,
where the second argument is a flag of type `Val{true}` or `Val{fals}` that indicates
whether the adjoint or the normal action of the linear map needs to be computed. The latter
form still combines well with the `do` block syntax of Julia, as in

```julia
vals, lvecs, rvecs, info = bieigsolve(x₀, howmany, which; kwargs...) do x, flag
    if flag === Val(true)
        # y = compute action of adjoint map on x
    else
        # y = compute action of linear map on x
    end
    return y
end
```

If an `AbstractMatrix` is used, a left (right) starting vector `x₀` (`y₀`) does not need to
be provided, it is then chosen as `rand(T, size(A, 1))` (`rand(T, size(A, 2))`). If the
linear map is encoded more generally as a callable function or method, the best approach is
to provide an explicit starting guess for `x₀` and `y₀`. Note that these do not need to be
of type `AbstractVector`; any type that behaves as a vector and supports the required
methods (see KrylovKit docs) is accepted. If instead an integer `n` is specified, it is
assumed that `x₀` and `y₀` are regular vectors and they are initialized to `rand(T, n)`,
where the default value of `T` is `Float64`, unless specified differently.

There are two optional keyword arguments that should typically be specified. `howmany`
specifies how many eigenvalues should be computed; `which` specifies which eigenvalues
should be targeted. Valid specifications of `which` are given by

  - `:LM`: eigenvalues of largest magnitude
  - `:LR`: eigenvalues with largest (most positive) real part
  - `:SR`: eigenvalues with smallest (most negative) real part
  - `:LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
  - `:SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
  - [`EigSorter(f; rev = false)`](@ref): eigenvalues `λ` that appear first (or last if
    `rev == true`) when sorted by `f(λ)`

!!! note "Note about selecting `which` eigenvalues"

    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. All of they valid `Symbol`s for `which` have this
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

The return value is always of the form `vals, lvecs, rvecs, info = eigsolve(...)` with

  - `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could
    be longer if more eigenvalues were converged at the same cost. Eigenvalues will be real
    if [`Lanczos`](@ref) was used and complex if [`Arnoldi`](@ref) was used (see below).
  - `lvecs` and `rvecs`: two `Vector`s of corresponding left and right eigenvectors, of the
    same length as `vals`. Note that eigenvectors are not returned as a matrix, as the
    linear map could act on any custom Julia type with vector like behavior, i.e. the
    elements of the list `lvecs` and `rvecs` are objects that are typically similar to the
    starting guesses `x₀` and `y₀`, up to a possibly different `eltype`. In particular for
    a general matrix (i.e. with `Arnoldi`) the eigenvectors are generally complex and are
    therefore always returned in a complex number format. When the linear map is a simple
    `AbstractMatrix`, `lvecs` and `rvecs` will be `Vector{Vector{<:Number}}`s.
  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: indicates how many eigenvalues and eigenvectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
      + `info.residual::Vector`: a list of twice the length of `vals` containing the
        residuals `info.residual[i] = fᴴ(vecs[i]) - conj(vals[i]) * vecs[i]` when `i <= length(vals)` and `info.residual[i] = f(vecs[i]) - vals[i] * vecs[i]` when `i > length(vals)`.
      + `info.normres::Vector{<:Real}`: list of twice the length of `vals` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A` or `A'`.
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

### Algorithm

The final (expert) method, without default values and keyword arguments, is the one that is
finally called, and can also be used directly. Here, one specifies the algorithm explicitly
as either [`Lanczos`](@ref), for real symmetric or complex hermitian problems, or
[`Arnoldi`](@ref), for general problems. Note that these names refer to the process for
building the Krylov subspace, but the actual algorithm is an implementation of the
Krylov-Schur algorithm, which can dynamically shrink and grow the Krylov subspace, i.e. the
restarts are so-called thick restarts where a part of the current Krylov subspace is kept.

When the algorithm is specified as [`Lanczos`](@ref), only right eigenvectors will be
computed, while the left eigenvectors will be a direct copy. For [`Arnoldi`](@ref), the
eigenvectors are computed separately.

!!! note "Note about convergence"

    In case of a general problem, where the `Arnoldi` method is used, convergence of an
    eigenvalue is not based on the norm of the residual `norm(f(vecs[i]) - vals[i]*vecs[i])`
    for the eigenvector but rather on the norm of the residual for the corresponding Schur
    vectors.

    See also [`schursolve`](@ref) if you want to use the partial Schur decomposition
    directly, or if you are not interested in computing the eigenvectors, and want to work
    in real arithmetic all the way true (if the linear map and starting guess are real).
"""
function bieigsolve end

bieigsolve(A::AbstractMatrix, T::Type = eltype(A); kwargs...) = 
    bieigsolve(A, rand(T, size(A, 1)), rand(T, size(A, 1)); kwargs...)

bieigsolve(f, n::Int, T::Type = Float64; kwargs...) = 
    bieigsolve(f, rand(T, n), rand(T, n); kwargs...)

function bieigsolve(f, x₀, y₀ = x₀; howmany::Int = 1, which::Selector = :LM, kwargs...)
    Tx = typeof(x₀)
    Tfx = Core.Compiler.return_type(apply, Tuple{typeof(f),Tx})
    T = Core.Compiler.return_type(dot, Tuple{Tx,Tfx})
    alg = eigselector(f, T; kwargs...)
    checkwhich(which) || error("Unknown eigenvalue selector: which = $which")
    if alg isa Lanczos
        if which == :LI || which == :SI
            error(
                "Eigenvalue selector which = $which invalid: real eigenvalues expected with Lanczos algorithm"
            )
        end
    elseif T <: Real
        if which == :LI ||
            which == :SI ||
            (which isa EigSorter && which.by(+im) != which.by(-im))
            error(
                "Eigenvalue selector which = $which invalid because it does not treat
          `λ` and `conj(λ)` equally: work in complex arithmetic by providing a complex starting vector `x₀`"
            )
        end
    end
    return bieigsolve(f, x₀, y₀, alg; which = which, howmany = howmany)
end
