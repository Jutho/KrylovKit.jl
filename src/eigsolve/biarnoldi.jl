"""
    bieigsolve(A::AbstractMatrix, [howmany = 1, which = :LM, T = eltype(A)]; kwargs...))
    bieigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    bieigsolve(f, v₀, w₀, [howmany = 1, which = :LM]; kwargs...)
    bieigsolve(f, v₀, w₀, howmany, which, alg::BiArnoldi)

Compute at least `howmany` eigenvalues from the linear map encoded in the matrix `A` or by the function `f`. Return eigenvalues, left- and right-eigenvectors and a `ConvergenceInfo` structure.

### Arguments:

The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. Since both the action of the linear map and its adjoint are required in
order to compute right- and left-eigenvectors, `f` can either be a tuple of two callable objects (each
accepting a single argument), representing the linear map and its adjoint respectively, or,
`f` can be a single callable object that accepts two input arguments, where the second
argument is a flag of type `Val{true}` or `Val{false}` that indicates whether the adjoint or
the normal action of the linear map needs to be computed. This is the same implementation as 
for `svd_solve`.

If the linear map is encoded more
generally as a a callable function or method, the best approach is to provide an explicit
starting guess `v₀` and `w₀` for  the right- and left-eigenvector. Note that `v₀` and `w₀` do not need to be of type `AbstractVector`; any type
that behaves as a vector and supports the required methods (see KrylovKit docs) is accepted.
If instead of `v₀` and `w₀` an integer `n` is specified, it is assumed that both are a regular vector
and they are initialized to `rand(T, n)`, where the default value of `T` is `Float64`, unless
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

### Return Values

Returns a tuple `(values, (vectorsV, vectorsW), (infoV, infoW))` where:

- `values`: Vector of computed eigenvalues.
- `vectorsV`: Vector of right eigenvectors (corresponding to `A`).
- `vectorsW`: Vector of left eigenvectors (corresponding to `A'`).
- `infoV`, `infoW`: `ConvergenceInfo` objects for right and left eigenproblems, containing:
    - `converged`: Number of converged eigenvalues.
    - `residual`: List of residual vectors.
    - `normres`: List of residual norms.
    - `numiter`: Number of Krylov restarts.
    - `numops`: Number of operator applications.

### Notes

- The BiArnoldi method is designed for non-Hermitian problems and returns both left and right eigenvectors, which are biorthogonal.
- For Hermitian or symmetric problems, prefer using `eigsolve` or `realeigsolve`.


!!! warning "Check for convergence"

    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Keyword arguments:

Keyword arguments and their default values are given by:

  - `verbosity::Int = SILENT_LEVEL`: verbosity level, i.e. 
    - SILENT_LEVEL (suppress all messages)
    - WARN_LEVEL (only warnings)
    - STARTSTOP_LEVEL (one message with convergence info at the end)
    - EACHITERATION_LEVEL (progress info after every iteration)
    - EACHITERATION_LEVEL+ (all of the above and additional information about the Lanczos, BlockLanczos, or Arnoldi iteration)
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

The default values are given by `tol = KrylovDefaults.tol`,
`krylovdim = KrylovDefaults.krylovdim`, `maxiter = KrylovDefaults.maxiter`,
`orth = KrylovDefaults.orth`; see [`KrylovDefaults`](@ref) for details.


### Example

```julia
A = randn(10, 10)
vals, (vecsR, vecsL), (infoR, infoL) = bieigsolve(A, 3, :LM)
```

See also: [`eigsolve`](@ref), [`realeigsolve`](@ref), [`BiArnoldi`](@ref)
"""
function bieigsolve end

function bieigsolve(
        A::AbstractMatrix, howmany::Int = 1, which::Selector = :LM, T::Type = eltype(A);
        kwargs...
    )
    v₀ = Random.rand!(similar(A, T, size(A, 1)))
    w₀ = Random.rand!(similar(A, T, size(A, 1)))
    return bieigsolve(A, v₀, w₀, howmany, which; kwargs...)
end

function bieigsolve(
        f, n::Int, howmany::Int = 1, which::Selector = :LM, T::Type = Float64;
        kwargs...
    )
    return bieigsolve(f, rand(T, n), rand(T, n), howmany, which; kwargs...)
end

function bieigsolve(
        f, v₀, w₀, howmany::Int = 1, which::Selector = :LM;
        krylovdim::Int = KrylovDefaults.krylovdim[],
        maxiter::Int = KrylovDefaults.maxiter[],
        eager::Bool = false,
        tol::Real = KrylovDefaults.tol[],
        orth::Orthogonalizer = KrylovDefaults.orth,
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    return bieigsolve(
        f, v₀, w₀, howmany, which,
        BiArnoldi(; orth, krylovdim, maxiter, tol, eager, verbosity)
    )
end

function bieigsolve(f, v₀, w₀, howmany::Int, which::Selector, alg::BiArnoldi)
    #! format: off
    (S, T), (Q, Z), (V, W), (rV, rW), (h, k), M, converged, numiter, numops =
        _bischursolve(f, v₀, w₀, howmany, which, alg)
    #! format: on

    howmany′ = howmany
    if eltype(T) <: Real && howmany < size(T, 1) && !iszero(T[howmany + 1, howmany])
        howmany′ += 1
    elseif size(T, 1) < howmany
        howmany′ = size(T, 1)
    end
    if converged > howmany
        howmany′ = converged
    end

    # Compute the eigenvalues and eigenvectors of the reduced matrices
    SS = view(S, 1:howmany′, 1:howmany′)
    TT = view(T, 1:howmany′, 1:howmany′)
    valuesS = schur2eigvals(SS)
    vecsS = schur2eigvecs(SS)

    # Instead of computing the eigenvectors of TT separately, we can use the
    # relation ZᴴMQ * S = T' * ZᴴMQ to compute vecsT directly from vecsS.
    # In this way, we avoid the potential ordering mismatch, and the resulting
    # left and right eigenvectors will automatically be biorthogonal.
    # Note that this requires that ZᴴMQ can be accurately inverted.
    ZᴴMQ = view(Z, :, 1:howmany′)' * M * view(Q, :, 1:howmany′)
    vecsT = inv(adjoint(ZᴴMQ * vecsS))
    valuesT = conj(valuesS)

    if !isapprox(TT * vecsT, vecsT * Diagonal(valuesT)) && alg.verbosity >= WARN_LEVEL
        @warn """Unexpected relation between S and T in BiArnoldi eigsolve:
        left eigenvectors might not be correctly computed
        """
    end

    # Construct the actual eigenvectors and residuals
    VS = view(Q, :, 1:howmany′) * vecsS
    vectorsS = [V * v for v in cols(VS)]
    hᴴVS = map(Base.Fix1(dot, h[1:howmany′]), cols(vecsS))
    residualsS = [scale(rV, s) for s in hᴴVS]
    normresidualsS = map(Base.Fix1(*, norm(rV)) ∘ abs, hᴴVS)
    VT = view(Z, :, 1:howmany′) * vecsT
    vectorsT = [W * v for v in cols(VT)]
    kᴴVT = map(Base.Fix1(dot, k[1:howmany′]), cols(vecsT))
    residualsT = [scale(rW, s) for s in kᴴVT]
    normresidualsT = map(Base.Fix1(*, norm(rW)) ∘ abs, kᴴVT)

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """BiArnoldi eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresidualsS))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """BiArnoldi eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresidualsS))
        * number of operations = $numops"""
    end

    infoS = ConvergenceInfo(converged, residualsS, normresidualsS, numiter, numops)
    infoT = ConvergenceInfo(converged, residualsT, normresidualsT, numiter, numops)

    return valuesS, (vectorsS, vectorsT), (infoS, infoT)
end

function _bischursolve(f, v₀, w₀, howmany::Int, which::Selector, alg::BiArnoldi)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize arnoldi factorization
    iter = BiArnoldiIterator(f, v₀, w₀, alg.orth)
    fact = initialize(iter; verbosity = alg.verbosity)
    numops = 1
    sizehint!(fact, krylovdim)
    βv, βw = normres(fact)
    tol::eltype(βv) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    KK = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    QQ = fill(zero(eltype(fact)), krylovdim, krylovdim)
    ZZ = fill(zero(eltype(fact)), krylovdim, krylovdim)
    Wᴴvv = fill(zero(eltype(fact)), krylovdim)
    Vᴴww = fill(zero(eltype(fact)), krylovdim)

    MM = fill(zero(eltype(fact)), krylovdim, krylovdim)
    MMQQ = fill(zero(eltype(fact)), krylovdim, krylovdim)
    ZZMMQQ = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize storage
    L = length(fact) # == 1
    V, W = basis(fact)
    MM[L, L] = inner(W[L], V[L])
    converged = 0
    local S, T, Q, Z, rV, rW, h, k, M
    while true
        βv, βw = normres(fact)
        L = length(fact)

        if (βv <= tol || βw <= tol) && L < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $L (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end
        end
        if L == krylovdim || (βv <= tol && βw <= tol) || (alg.eager && L >= howmany) # process
            # Assign storage as views of the allocated arrays
            H = view(HH, 1:L, 1:L)
            K = view(KK, 1:L, 1:L)
            Q = view(QQ, 1:L, 1:L)
            Z = view(ZZ, 1:L, 1:L)
            M = view(MM, 1:L, 1:L)
            h = view(HH, L + 1, 1:L)
            k = view(KK, L + 1, 1:L)
            Wᴴv = view(Wᴴvv, 1:L)
            Vᴴw = view(Vᴴww, 1:L)
            MQ = view(MMQQ, 1:L, 1:L)
            ZMQ = view(ZZMMQQ, 1:L, 1:L)

            # Initialize values
            copyto!(Q, I)
            copyto!(Z, I)
            copyto!.((H, K), rayleighquotient(fact))
            # h .= SimpleBasisVector(L, L)
            # k .= SimpleBasisVector(L, L)

            # Step 1 - Normalize residuals so that they represent next Arnoldi vectors
            rV, rW = residual(fact)
            if L < krylovdim
                # explicitly copy the residual to not disturb future expansion steps
                rV = scale(rV, 1 / βv) # vℓ₊₁
                rW = scale(rW, 1 / βw) # wℓ₊₁
            else
                rV = scale!!(rV, 1 / βv) # vℓ₊₁
                rW = scale!!(rW, 1 / βw) # wℓ₊₁
            end
            # we remember the value of h and k until the first time we actually need to construt them
            # h .*= βv # or thus: h = βv * SimpleBasisVector(L, L)
            # k .*= βw # or thus: k = βw * SimpleBasisVector(L, L)

            # Step 2 - 3 - Correct H, K and the residuals using the oblique projection
            # Compute the projections Wᴴ * residual(V) and Vᴴ * residual(W)
            V, W = basis(fact)
            @inbounds for i in 1:L
                Wᴴv[i] = inner(W[i], rV)
                Vᴴw[i] = inner(V[i], rW)
            end

            luM = lu(M)
            M⁻¹Wᴴv = luM \ Wᴴv
            M⁻ᴴVᴴw = luM' \ Vᴴw
            add!(view(H, :, L), M⁻¹Wᴴv, βv) # H̃ = H + (M \ (W' * v)) * h'
            add!(view(K, :, L), M⁻ᴴVᴴw, βw) # K̃ = K + (M' \ (V' * w)) * k'

            @inbounds for i in 1:L
                rV = add!!(rV, V[i], -M⁻¹Wᴴv[i]) # ṽℓ₊₁
                rW = add!!(rW, W[i], -M⁻ᴴVᴴw[i]) # w̃ℓ₊₁
            end

            βrV = norm(rV) # || ṽℓ₊₁ ||
            βrW = norm(rW) # || w̃ℓ₊₁ ||

            # Step 5 - Compute dense schur factorization
            S, Q, valuesH = hschur!(H, Q)
            T, Z, valuesK = hschur!(K, Z)

            # Step 6 - Order the Schur decompositions
            by, rev = eigsort(which)
            pH = sortperm(valuesH; by = by, rev = rev)
            pK = sortperm(valuesK; by = by ∘ conj, rev = rev)

            S, Q = permuteschur!(S, Q, pH)
            T, Z = permuteschur!(T, Z, pK)

            # Partially Step 7 & 8 - Correction of hm and km
            h .= conj.(view(Q, L, :)) .* βv # h̃ = Q' * h
            k .= conj.(view(Z, L, :)) .* βw # k̃ = Z' * k

            # At this point, we have the partial Schur decompositions of the form
            # A * V * Q = V * Q * S + rV * h'
            # A' * W * Z = W * Z * T + rW * k'
            # with W' * rV = 0 and V' * rW = 0

            converged = 0
            while converged < length(fact)
                # As in the Arnoldi case, we will not actually compute the
                # eigenvectors yet, but use the "residuals" of the Schur vectors
                # to determine how many vectors have converged.
                xh = βrV * abs(h[converged + 1])
                xk = βrW * abs(k[converged + 1])
                if max(xh, xk) <= tol
                    converged += 1
                else
                    break
                end
            end
            if 0 < converged < length(fact) && !iszero(S[converged + 1, converged])
                converged -= 1
            end

            if converged >= howmany || (βv <= tol && βw <= tol)
                break
            elseif alg.verbosity >= EACHITERATION_LEVEL
                @info "BiArnoldi schursolve in iteration $numiter, step = $L: $converged values converged, normres = $(normres2string(abs.(h[1:howmany])))"
            end
        end

        if L < krylovdim # expand
            fact = expand!(iter, fact; verbosity = alg.verbosity)
            V, W = basis(fact)

            # update M with the new basis vectors
            @inbounds for i in 1:L
                MM[i, L + 1] = inner(W[i], V[L + 1])
                MM[L + 1, i] = inner(W[L + 1], V[i])
            end
            MM[L + 1, L + 1] = inner(W[L + 1], V[L + 1])

            numops += 2
        else # shrink
            numiter == maxiter && break

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            while (!iszero(H[keep + 1, keep]) || !iszero(K[keep + 1, keep]))
                # we are in the middle of a 2x2 block; this cannot happen if keep == converged, so we can decrease keep
                # however, we have to make sure that we do not end up with keep = 0
                if keep > 1
                    keep -= 1 # conservative choice
                else
                    keep += 1
                    if krylovdim == 2
                        alg.verbosity >= WARN_LEVEL &&
                            @warn "Arnoldi iteration got stuck in a 2x2 block, consider increasing the Krylov dimension"
                        break
                    end
                end
            end

            # Step 9 to 11 - Shrink Krylov factorization and restore Arnoldi form for both directions
            # In particular, we need to return to a Rayleigh quotient that is `V' * A * V` instead
            # of `W' * A * V`, and residuals that satisfy `V' * rV = 0` and `W' * rW = 0` instead of
            # `W' * rV = 0` and `V' * rW = 0`.

            # Step 10 and 11: update Rayleigh quotients and residuals
            # Ĥ = Sₖₖ + (V * Qₖ)' * ṽℓ₊₁ * h̃ₖ' (subscript k for `keep`)
            # and
            # (V * Qₖ)' * ṽℓ₊₁ = Qₖ' * V' * (vℓ₊₁ - V * M \ (W' * vℓ₊₁))= - Qₖ' * M \ (W' * vℓ₊₁)
            # so that (and analoguously for K)
            # Ĥ = Sₖₖ - Qₖ' * M⁻¹Wᴴv * h̃'ₖ
            # K̂ = Tₖₖ - Zₖ' * M⁻ᴴVᴴw * k̃'ₖ

            VQᴴv = -view(Q, :, 1:keep)' * M⁻¹Wᴴv
            WZᴴw = -view(Z, :, 1:keep)' * M⁻ᴴVᴴw

            H[1:keep, 1:keep] += VQᴴv * h[1:keep]' # Ĥ = Sₖₖ + (V * Qₖ)' * ṽℓ₊₁ * h̃ₖ'
            K[1:keep, 1:keep] += WZᴴw * k[1:keep]' # K̂ = Tₖₖ + (W * Zₖ)' * w̃ℓ₊₁ * k̃ₖ'

            # We similarly correct the residuals
            # v̂ = ṽℓ₊₁ - (V * Qₖ) * (V * Qₖ)' * ṽℓ₊₁
            #   = ṽℓ₊₁ + (V * Qₖ) * (V * Qₖ)' * V * M \ (W' * vℓ₊₁)
            #   = ṽℓ₊₁ + V * Qₖ * Qₖ' * M⁻¹Wᴴv
            #   = ṽℓ₊₁ - V * Qₖ * VQᴴv
            # Let's also recylce the Wᴴv and Vᴴw storage, which we don't need anymore
            QVQᴴv = mul!(Wᴴv, view(Q, :, 1:keep), VQᴴv)
            ZWZᴴw = mul!(Vᴴw, view(Z, :, 1:keep), WZᴴw)
            @inbounds for i in 1:L
                rV = add!!(rV, V[i], -QVQᴴv[i]) # v̂ = ṽℓ₊₁ - V * Qₖ * VQᴴv
                rW = add!!(rW, W[i], -ZWZᴴw[i]) # ŵ = w̃ℓ₊₁ - W * Zₖ * WZᴴw
            end

            # normalize the new residuals and absorb norm in h and k
            βrV = norm(rV)
            βrW = norm(rW)
            rV = scale!!(rV, 1 / βrV) # v̂ₖ₊₁
            rW = scale!!(rW, 1 / βrW) # ŵₖ₊₁
            h .*= βrV
            k .*= βrW

            # Restore Arnoldi form in the first keep columns before shrinking
            _restorearnoldiform!(Q, H, h, keep)
            _restorearnoldiform!(Z, K, k, keep)
            # Copy H and K back into compact Hessenberg form
            copy!.(rayleighquotient(fact), (H, K))
            # Update the basis
            basistransform!(V, view(Q, :, 1:keep))
            V[keep + 1] = rV
            basistransform!(W, view(Z, :, 1:keep))
            W[keep + 1] = rW

            # Step 9: update M; we can only do this now because by restoring the Arnoldi form where
            # the Rayleigh quotients have a Hessenberg form, we have further changed `Q` and `Z` and
            # thus the actual new Krylov bases `V` and `W`.
            MQ = view(MMQQ, 1:L, 1:keep)
            ZMQ = view(ZZMMQQ, 1:keep, 1:keep)
            MQ = mul!(MQ, M, view(Q, :, 1:keep))
            ZMQ = mul!(ZMQ, view(Z, :, 1:keep)', MQ)
            copy!(view(MM, 1:keep, 1:keep), ZMQ)

            # Shrink BiArnoldi factorization
            fact = shrink!(fact, keep; verbosity = alg.verbosity)
            numiter += 1
        end
    end

    # At the point where we return, `fact` is not in a valid state, as we have the
    # Krylov factorization or partial Schur decomposition in the form
    # A * V * Q = V * Q * S + rV * h'
    # A' * W * Z = W * Z * T + rW * k'
    # with W' * rV = 0 and V' * rW = 0, so this is the information we return.
    # We also return M = W' * V.
    return (S, T), (Q, Z), (V, W), (rV, rW), (h, k), M, converged, numiter, numops
end
