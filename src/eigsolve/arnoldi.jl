"""
    # expert version:
    schursolve(f, x₀, howmany, which, algorithm)

Compute a partial Schur decomposition containing `howmany` eigenvalues from the linear map
encoded in the matrix or function `A`. Return the reduced Schur matrix, the basis of Schur
vectors, the extracted eigenvalues and a `ConvergenceInfo` structure.

See also [`eigsolve`](@ref) to obtain the eigenvectors instead. For real symmetric or
complex hermitian problems, the (partial) Schur decomposition is identical to the (partial)
eigenvalue decomposition, and `eigsolve` should always be used.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object, that acts on vector like objects similar to `x₀`, which is the starting
guess from which a Krylov subspace will be built. `howmany` specifies how many Schur vectors
should be converged before the algorithm terminates; `which` specifies which eigenvalues
should be targeted. Valid specifications of `which` are

  - `LM`: eigenvalues of largest magnitude
  - `LR`: eigenvalues with largest (most positive) real part
  - `SR`: eigenvalues with smallest (most negative) real part
  - `LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
  - `SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
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

The `algorithm` argument currently only supports an instance of [`Arnoldi`](@ref), which
is where the parameters of the Krylov method (such as Krylov dimension and maximum number
of iterations) can be specified. Since `schursolve` is less commonly used as `eigsolve`,
it only supports this expert mode call syntax and no convenient keyword interface is
currently available.

### Return values:

The return value is always of the form `T, vecs, vals, info = schursolve(...)` with

  - `T`: a `Matrix` containing the partial Schur decomposition of the linear map, i.e. it's
    elements are given by `T[i,j] = dot(vecs[i], f(vecs[j]))`. It is of Schur form, i.e.
    upper triangular in case of complex arithmetic, and block upper triangular (with at most
    2x2 blocks) in case of real arithmetic.
  - `vecs`: a `Vector` of corresponding Schur vectors, of the same length as `vals`. Note
    that Schur vectors are not returned as a matrix, as the linear map could act on any
    custom  Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`, up to a possibly
    different `eltype`. When the linear map is a simple `AbstractMatrix`, `vecs` will be
    `Vector{Vector{<:Number}}`. Schur vectors are by definition orthogonal, i.e.
    `dot(vecs[i],vecs[j]) = I[i,j]`. Note that Schur vectors are real if the problem (i.e.
    the linear map and the initial guess) are real.
  - `vals`: a `Vector` of eigenvalues, i.e. the diagonal elements of `T` in case of complex
    arithmetic, or extracted from the diagonal blocks in case of real arithmetic. Note that
    `vals` will always be complex, independent of the underlying arithmetic.
  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: indicates how many eigenvalues and Schur vectors were
        actually converged to the specified tolerance (see below under keyword arguments)

      + `info.residuals::Vector`: a list of the same length as `vals` containing the actual
        residuals

        ```julia
        info.residuals[i] = f(vecs[i]) - sum(vecs[j] * T[j, i] for j in 1:i+1)
        ```

        where `T[i+1,i]` is definitely zero in case of complex arithmetic and possibly zero
        in case of real arithmetic
      + `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual for every Schur vector, i.e.
        `info.normes[i] = norm(info.residual[i])`
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
      + `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

!!! warning "Check for convergence"

    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Algorithm

The actual algorithm is an implementation of the Krylov-Schur algorithm, where the
[`Arnoldi`](@ref) algorithm is used to generate the Krylov subspace. During the algorithm,
the Krylov subspace is dynamically grown and shrunk, i.e. the restarts are so-called thick
restarts where a part of the current Krylov subspace is kept.
"""
function schursolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    T, U, fact, converged, numiter, numops = _schursolve(A, x₀, howmany, which, alg)
    if eltype(T) <: Real && howmany < length(fact) && T[howmany + 1, howmany] != 0
        howmany += 1
    end
    if converged > howmany
        howmany = converged
    end
    TT = view(T, 1:howmany, 1:howmany)
    values = schur2eigvals(TT)
    vectors = let B = basis(fact)
        [B * u for u in cols(U, 1:howmany)]
    end
    residuals = let r = residual(fact)
        [scale(r, last(u)) for u in cols(U, 1:howmany)]
    end
    normresiduals = [normres(fact) * abs(last(u)) for u in cols(U, 1:howmany)]

    if alg.verbosity > 0
        if converged < howmany
            @warn """Arnoldi schursolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))"""
        else
            @info """Arnoldi schursolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))"""
        end
    end
    return TT,
           vectors,
           values,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi; alg_rrule=alg)
    T, U, fact, converged, numiter, numops = _schursolve(A, x₀, howmany, which, alg)
    if eltype(T) <: Real && howmany < length(fact) && T[howmany + 1, howmany] != 0
        howmany += 1
    end
    if converged > howmany
        howmany = converged
    end
    d = min(howmany, size(T, 2))
    TT = view(T, 1:d, 1:d)
    values = schur2eigvals(TT)

    # Compute eigenvectors
    V = view(U, :, 1:d) * schur2eigvecs(TT)
    vectors = let B = basis(fact)
        [B * v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [scale(r, last(v)) for v in cols(V)]
    end
    normresiduals = [normres(fact) * abs(last(v)) for v in cols(V)]

    if alg.verbosity > 0
        if converged < howmany
            @warn """Arnoldi eigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Arnoldi eigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end
    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

"""
    # expert version:
    realeigsolve(f, x₀, howmany, which, algorithm; alg_rrule=algorithm)

Compute the first `howmany` eigenvalues (according to the order specified by `which`)
from the real linear map encoded in the matrix `A` or by the function `f`, if it can be
guaranteed that these eigenvalues (and thus their associated eigenvectors) are real. An
error will be thrown if there are complex eigenvalues within the first `howmany` eigenvalues.

Return eigenvalues, eigenvectors and a `ConvergenceInfo` structure.

!!! note "Note about real linear maps"

    A function `f` is said to implement a real linear map if it satisfies 
    `f(add(x,y)) = add(f(x), f(y)` and `f(scale(x, α)) = scale(f(x), α)` for vectors `x`
    and `y` and scalars `α::Real`. Note that this is possible even when the vectors are
    represented using complex arithmetic. For example, the map `f=x-> x + conj(x)`
    represents a real linear map that is not (complex) linear, as it does not satisfy
    `f(scale(x, α)) = scale(f(x), α)` for complex scalars `α`. Note that complex linear
    maps are always real linear maps and thus can be used in this context, if looking
    specifically for real eigenvalues that they may have.

    To interpret the vectors `x` and `y` as elements from a real vector space, the standard
    inner product defined on them will be replaced with `real(inner(x,y))`. This has no
    effect if the vectors `x` and `y` were represented using real arithmetic to begin with,
    and allows to seemlessly use complex vectors as well.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. A starting vector `x₀` needs to be provided. Note that `x₀` does not need
to be of type `AbstractVector`; any type that behaves as a vector and supports the required
interface (see KrylovKit docs) is accepted.

The argument `howmany` specifies how many eigenvalues should be computed; `which` specifies
which eigenvalues should be targeted. Valid specifications of `which` for real
problems are given by

  - `:LM`: eigenvalues of largest magnitude
  - `:LR`: eigenvalues with largest (most positive) real part
  - `:SR`: eigenvalues with smallest (most negative) real part
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

The `algorithm` argument currently only supports an instance of [`Arnoldi`](@ref), which
is where the parameters of the Krylov method (such as Krylov dimension and maximum number
of iterations) can be specified. Since `realeigsolve` is less commonly used as `eigsolve`,
it only supports this expert mode call syntax and no convenient keyword interface is
currently available.

The keyword argument `alg_rrule` can be used to specify an algorithm to be used for computing
the `pullback` of `realeigsolve` in the context of reverse-mode automatic differentation.
    
### Return values:

The return value is always of the form `vals, vecs, info = eigsolve(...)` with

  - `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could
    be longer if more eigenvalues were converged at the same cost. Eigenvalues will be real,
    an `ArgumentError` will be thrown if the first `howmany` eigenvalues ordered according
    to `which` of the linear map are not all real.
  - `vecs`: a `Vector` of corresponding eigenvectors, of the same length as `vals`. Note
    that eigenvectors are not returned as a matrix, as the linear map could act on any
    custom Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`. For a real problem with
    real eigenvalues, also the eigenvectors will be real and no complex arithmetic is used
    anywhere.
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
"""
function realeigsolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi; alg_rrule=alg)
    T, U, fact, converged, numiter, numops = _schursolve(A, RealVec(x₀), howmany, which,
                                                         alg)
    i = 0
    while i < length(fact)
        i += 1
        if i < length(fact) && T[i + 1, i] != 0
            i -= 1
            break
        end
    end
    i < howmany &&
        throw(ArgumentError("only the first $i eigenvalues are real, which is less then the requested `howmany = $howmany`"))
    howmany = max(howmany, min(i, converged))
    TT = view(T, 1:howmany, 1:howmany)
    values = diag(TT)

    # Compute eigenvectors
    V = view(U, :, 1:howmany) * schur2realeigvecs(TT)
    vectors = let B = basis(fact)
        [(B * v)[] for v in cols(V)]
    end
    residuals = let r = residual(fact)[]
        [scale(r, last(v)) for v in cols(V)]
    end
    normresiduals = [normres(fact) * abs(last(v)) for v in cols(V)]

    if alg.verbosity > 0
        if converged < howmany
            @warn """Arnoldi realeigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Arnoldi realeigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end
    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function _schursolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity - 2)
    numops = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize storage
    K = length(fact) # == 1
    converged = 0
    local T, U
    while true
        β = normres(fact)
        K = length(fact)

        if β <= tol
            if K < howmany
                @warn "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`); setting `howmany = $K`."
                howmany = K
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && K >= howmany) # process
            H = view(HH, 1:K, 1:K)
            U = view(UU, 1:K, 1:K)
            f = view(HH, K + 1, 1:K)
            copyto!(U, I)
            copyto!(H, rayleighquotient(fact))

            # compute dense schur factorization
            T, U, values = hschur!(H, U)
            by, rev = eigsort(which)
            p = sortperm(values; by=by, rev=rev)
            T, U = permuteschur!(T, U, p)
            f = mul!(f, view(U, K, :), β)
            converged = 0
            while converged < length(fact) && abs(f[converged + 1]) <= tol
                converged += 1
            end
            if eltype(T) <: Real &&
               0 < converged < length(fact) &&
               T[converged + 1, converged] != 0
                converged -= 1
            end

            if converged >= howmany
                break
            elseif alg.verbosity > 1
                msg = "Arnoldi schursolve in iter $numiter, krylovdim = $K: "
                msg *= "$converged values converged, normres = ("
                msg *= @sprintf("%.2e", abs(f[1]))
                for i in 2:howmany
                    msg *= ", "
                    msg *= @sprintf("%.2e", abs(f[i]))
                end
                msg *= ")"
                @info msg
            end
        end

        if K < krylovdim # expand
            fact = expand!(iter, fact; verbosity=alg.verbosity - 2)
            numops += 1
        else # shrink
            numiter == maxiter && break

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
            if eltype(H) <: Real && H[keep + 1, keep] != 0 # we are in the middle of a 2x2 block
                keep += 1 # conservative choice
                keep >= krylovdim &&
                    error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
            end

            # Restore Arnoldi form in the first keep columns
            @inbounds for j in 1:keep
                H[keep + 1, j] = f[j]
            end
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + 1, 1:j, j)
                H[j + 1, j] = ν
                H[j + 1, 1:(j - 1)] .= 0
                lmul!(h, H)
                rmul!(view(H, 1:j, :), h')
                rmul!(U, h')
            end
            copyto!(rayleighquotient(fact), H) # copy back into fact

            # Update B by applying U
            B = basis(fact)
            basistransform!(B, view(U, :, 1:keep))
            r = residual(fact)
            B[keep + 1] = scale!!(r, 1 / normres(fact))

            # Shrink Arnoldi factorization
            fact = shrink!(fact, keep)
            numiter += 1
        end
    end
    return T, U, fact, converged, numiter, numops
end
