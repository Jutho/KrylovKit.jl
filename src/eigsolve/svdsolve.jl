"""
    svdsolve(A::AbstractMatrix, [x₀, howmany = 1, which = :LR, T = eltype(A)]; kwargs...)
    svdsolve(f, m::Int, [howmany = 1, which = :LR, T = Float64]; kwargs...)
    svdsolve(f, x₀, [howmany = 1, which = :LR]; kwargs...)
    # expert version:
    svdsolve(f, x₀, howmany, which, algorithm; alg_rrule=...)

Compute `howmany` singular values from the linear map encoded in the matrix `A` or by the
function `f`. Return singular values, left and right singular vectors and a
`ConvergenceInfo` structure.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. Since both the action of the linear map and its adjoint are required in
order to compute singular values, `f` can either be a tuple of two callable objects (each
accepting a single argument), representing the linear map and its adjoint respectively, or,
`f` can be a single callable object that accepts two input arguments, where the second
argument is a flag of type `Val{true}` or `Val{false}` that indicates whether the adjoint or
the normal action of the linear map needs to be computed. The latter form still combines
well with the `do` block syntax of Julia, as in

```julia
vals, lvecs, rvecs, info = svdsolve(x₀, howmany, which; kwargs...) do x, flag
    if flag === Val(true)
        # y = compute action of adjoint map on x
    else
        # y = compute action of linear map on x
    end
    return y
end
```

For a general linear map encoded using either the tuple or the two-argument form, the best
approach is to provide a start vector `x₀` (in the codomain, i.e. column space, of the
linear map). Alternatively, one can specify the number `m` of rows of the linear map, in
which case `x₀ = rand(T, m)` is used, where the default value of `T` is `Float64`, unless
specified differently. If an `AbstractMatrix` is used, a starting vector `x₀` does not need
to be provided; it is chosen as `rand(T, size(A, 1))`.

The next arguments are optional, but should typically be specified. `howmany` specifies how
many singular values and vectors should be computed; `which` specifies which singular
values should be targeted. Valid specifications of `which` are

  - `LR`: largest singular values
  - `SR`: smallest singular values
    However, the largest singular values tend to converge more rapidly.

### Return values:

The return value is always of the form `vals, lvecs, rvecs, info = svdsolve(...)` with

  - `vals`: a `Vector{<:Real}` containing the singular values, of length at least `howmany`,
    but could be longer if more singular values were converged at the same cost.
  - `lvecs`: a `Vector` of corresponding left singular vectors, of the same length as
    `vals`.
  - `rvecs`: a `Vector` of corresponding right singular vectors, of the same length as
    `vals`. Note that singular vectors are not returned as a matrix, as the linear map
    could act on any custom Julia type with vector like behavior, i.e. the elements of the
  lists `lvecs`(`rvecs`) are objects that are typically similar to the starting guess `x₀`(`A' * x₀`), up to a possibly different `eltype`. When the linear map is a simple
    `AbstractMatrix`, `lvecs` and `rvecs` will be `Vector{Vector{<:Number}}`.
  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: indicates how many singular values and vectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
      + `info.residual::Vector`: a list of the same length as `vals` containing the
        residuals
        `info.residual[i] = A * rvecs[i] - vals[i] * lvecs[i]`.
      + `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A` or `A'`.
      + `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

!!! warning "Check for convergence"

    No warning is printed if not all requested singular values were converged, so always
    check if `info.converged >= howmany`.

### Keyword arguments:

Keyword arguments and their default values are given by:

  - `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
  - `krylovdim`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should
    not necessarily support the `Base.length` function. If you know the actual problem
    dimension is smaller than the default value, it is useful to reduce the value of
    `krylovdim`, though in principle this should be detected.
  - `tol`: the requested accuracy according to `normres` as defined above. If you work in
    e.g. single precision (`Float32`), you should definitely change the default value.
  - `maxiter`: the number of times the Krylov subspace can be rebuilt; see below for further
    details on the algorithms.
  - `orth`: the orthogonalization method to be used, see [`Orthogonalizer`](@ref)
  - `eager::Bool = false`: if true, eagerly compute the SVD after every expansion of the
    Krylov subspace to test for convergence, otherwise wait until the Krylov subspace has
    dimension `krylovdim`

The final keyword argument `alg_rrule` is relevant only when `svdsolve` is used in a setting
where reverse-mode automatic differentation will be used. A custom `ChainRulesCore.rrule` is
defined for `svdsolve`, which can be evaluated using different algorithms that can be specified
via `alg_rrule`. A suitable default is chosen, so this keyword argument should only be used
when this default choice is failing or not performing efficiently. Check the documentation for
more information on the possible values for `alg_rrule` and their implications on the algorithm
being used.

### Algorithm

The last method, without default values and keyword arguments, is the one that is finally
called, and can also be used directly. Here the algorithm is specified, though currently
only [`GKL`](@ref) is available. `GKL` refers to the the partial Golub-Kahan-Lanczos
bidiagonalization which forms the basis for computing the approximation to the singular
values. This factorization is dynamically shrunk and expanded (i.e. thick restart) similar
to the Krylov-Schur factorization for eigenvalues.
"""
function svdsolve end

function svdsolve(A::AbstractMatrix,
                  howmany::Int=1,
                  which::Selector=:LR,
                  T::Type=eltype(A);
                  kwargs...)
    x₀ = Random.rand!(similar(A, T, size(A, 1)))
    return svdsolve(A, x₀, howmany, which; kwargs...)
end
function svdsolve(f, n::Int, howmany::Int=1, which::Selector=:LR, T::Type=Float64;
                  kwargs...)
    return svdsolve(f, rand(T, n), howmany, which; kwargs...)
end

function svdsolve(f, x₀, howmany::Int=1, which::Selector=:LR; kwargs...)
    which == :LR ||
        which == :SR ||
        error("invalid specification of which singular values to target: which = $which")
    alg = GKL(; kwargs...)
    return svdsolve(f, x₀, howmany, which, alg)
end

function svdsolve(A, x₀, howmany::Int, which::Symbol, alg::GKL;
                  alg_rrule=Arnoldi(; tol=alg.tol,
                                    krylovdim=alg.krylovdim,
                                    maxiter=alg.maxiter,
                                    eager=alg.eager,
                                    orth=alg.orth,
                                    verbosity=alg.verbosity))
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany singular values")

    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize GKL factorization
    iter = GKLIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity - 2)
    numops = 2
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::typeof(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    PP = fill(zero(eltype(fact)), krylovdim, krylovdim)
    QQ = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize storage
    local P, Q, f, S
    converged = 0
    while true
        β = normres(fact)
        K = length(fact)

        if β < tol
            if K < howmany
                @warn "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), which is smaller than the number of requested singular values (i.e. `howmany == $howmany`); setting `howmany = $K`."
                howmany = K
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && K >= howmany)
            P = copyto!(view(PP, 1:K, 1:K), I)
            Q = copyto!(view(QQ, 1:K, 1:K), I)
            f = view(HH, K + 1, 1:K)
            B = rayleighquotient(fact) # Bidiagional (lower)

            if K < krylovdim
                B = deepcopy(B)
            end
            P, S, Q = bidiagsvd!(B, P, Q)
            if which == :SR
                reversecols!(P)
                reverserows!(S)
                reverserows!(Q)
            elseif which != :LR
                error("invalid specification of which singular values to target: which = $which")
            end
            mul!(f, view(Q', K, :), β)

            converged = 0
            while converged < K && abs(f[converged + 1]) < tol
                converged += 1
            end

            if converged >= howmany
                break
            elseif alg.verbosity > 1
                msg = "GKL svdsolve in iter $numiter, krylovdim $krylovdim: "
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
            numops += 2
        else ## shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            # Update basis by applying P and Q using Householder reflections
            U = basis(fact, :U)
            basistransform!(U, view(P, :, 1:keep))
            # for j = 1:m
            #     h, ν = householder(P, j:m, j)
            #     lmul!(h, view(P, :, j+1:krylovdim))
            #     rmul!(U, h')
            # end
            V = basis(fact, :V)
            basistransform!(V, view(Q', :, 1:keep))
            # for j = 1:m
            #     h, ν = householder(Q, j, j:m)
            #     rmul!(view(Q, j+1:krylovdim, :), h)
            #     rmul!(V, h)
            # end

            # Shrink GKL factorization (no longer strictly GKL)
            r = residual(fact)
            U[keep + 1] = scale!!(r, 1 / normres(fact))
            H = fill!(view(HH, 1:(keep + 1), 1:keep), zero(eltype(HH)))
            @inbounds for j in 1:keep
                H[j, j] = S[j]
                H[keep + 1, j] = f[j]
            end

            # Restore bidiagonal form in the first keep columns
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + 1, 1:j, j)
                H[j + 1, j] = ν
                H[j + 1, 1:(j - 1)] .= zero(eltype(H))
                rmul!(view(H, 1:j, :), h')
                rmul!(V, h')
                h, ν = householder(H, 1:j, j, j)
                H[j, j] = ν
                @inbounds H[1:(j - 1), j] .= zero(eltype(H))
                lmul!(h, view(H, :, 1:(j - 1)))
                rmul!(U, h')
            end
            @inbounds for j in 1:keep
                fact.αs[j] = H[j, j]
                fact.βs[j] = H[j + 1, j]
            end
            # Shrink GKL factorization
            fact = shrink!(fact, keep)
            numiter += 1
        end
    end
    if converged > howmany
        howmany = converged
    end
    values = S[1:howmany]

    # Compute schur vectors
    Pv = view(P, :, 1:howmany)
    Qv = view(Q, 1:howmany, :)

    # Compute convergence information
    leftvectors = let U = basis(fact, :U)
        [U * v for v in cols(Pv)]
    end
    rightvectors = let V = basis(fact, :V)
        [V * v for v in cols(Qv')]
    end
    residuals = let r = residual(fact)
        [scale(r, last(v)) for v in cols(Qv')]
    end
    normresiduals = let f = f
        map(i -> abs(f[i]), 1:howmany)
    end
    if alg.verbosity > 0
        if converged < howmany
            @warn """GKL svdsolve finished without convergence after $numiter iterations:
             *  $converged singular values converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """GKL svdsolve finished after $numiter iterations:
             *  $converged singular values converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end

    return values,
           leftvectors,
           rightvectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
