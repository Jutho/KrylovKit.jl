function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos;
                  alg_rrule=Arnoldi(; tol=alg.tol,
                                    krylovdim=alg.krylovdim,
                                    maxiter=alg.maxiter,
                                    eager=alg.eager,
                                    orth=alg.orth))
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    if howmany > krylovdim
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end

    ## FIRST ITERATION: setting up
    # Initialize Lanczos factorization
    iter = LanczosIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity)
    numops = 1
    numiter = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::typeof(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    converged = 0
    local D, U, f
    while true
        β = normres(fact)
        K = length(fact)

        # diagonalize Krylov factorization
        if β <= tol && K < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && K >= howmany)
            U = copyto!(view(UU, 1:K, 1:K), I)
            f = view(HH, K + 1, 1:K)
            T = rayleighquotient(fact) # symtridiagonal

            # compute eigenvalues
            if K == 1
                D = [T[1, 1]]
                f[1] = β
                converged = Int(β <= tol)
            else
                if K < krylovdim
                    T = deepcopy(T)
                end
                D, U = tridiageigh!(T, U)
                by, rev = eigsort(which)
                p = sortperm(D; by=by, rev=rev)
                D, U = permuteeig!(D, U, p)
                mul!(f, view(U, K, :), β)
                converged = 0
                while converged < K && abs(f[converged + 1]) <= tol
                    converged += 1
                end
            end

            if converged >= howmany || β <= tol
                break
            elseif alg.verbosity >= EACHITERATION_LEVEL
                @info "Lanczos eigsolve in iteration $numiter, step = $K: $converged values converged, normres = $(normres2string(abs.(f[1:howmany])))"
            end
        end

        if K < krylovdim # expand Krylov factorization
            fact = expand!(iter, fact; verbosity=alg.verbosity)
            numops += 1
        else ## shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            # Restore Lanczos form in the first keep columns
            H = fill!(view(HH, 1:(keep + 1), 1:keep), zero(eltype(HH)))
            @inbounds for j in 1:keep
                H[j, j] = D[j]
                H[keep + 1, j] = f[j]
            end
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + 1, 1:j, j)
                H[j + 1, j] = ν
                H[j + 1, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:j, :), h')
                rmul!(U, h')
            end
            @inbounds for j in 1:keep
                fact.αs[j] = H[j, j]
                fact.βs[j] = H[j + 1, j]
            end

            # Update B by applying U using Householder reflections
            B = basis(fact)
            B = basistransform!(B, view(U, :, 1:keep))
            r = residual(fact)
            B[keep + 1] = scale!!(r, 1 / β)

            # Shrink Lanczos factorization
            fact = shrink!(fact, keep; verbosity=alg.verbosity)
            numiter += 1
        end
    end

    howmany′ = howmany
    if converged > howmany
        howmany′ = converged
    elseif length(D) < howmany
        howmany′ = length(D)
    end
    values = D[1:howmany′]

    # Compute eigenvectors
    V = view(U, :, 1:howmany′)

    # Compute convergence information
    vectors = let B = basis(fact)
        [B * v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [scale(r, last(v)) for v in cols(V)]
    end
    normresiduals = let f = f
        map(i -> abs(f[i]), 1:howmany′)
    end

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """Lanczos eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """Lanczos eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function eigsolve(A, x₀::T, howmany::Int, which::Selector, alg::BlockLanczos) where {T}
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    if howmany > krylovdim
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end
    tol = alg.tol
    verbosity = alg.verbosity

    # Initialize a block of vectors from the initial vector, randomly generated
    block0 = initialize(x₀, alg.blocksize)
    bs = length(block0)

    iter = BlockLanczosIterator(A, block0, krylovdim + bs, alg.qr_tol, alg.orth)
    fact = initialize(iter; verbosity=verbosity)  # Returns a BlockLanczosFactorization
    S = eltype(fact.TDB)  # The element type (Note: can be Complex) of the block tridiagonal matrix
    numops = 1    # Number of matrix-vector multiplications (for logging)
    numiter = 1

    converged = 0
    local normresiduals, D, U

    while true
        K = length(fact)
        β = normres(fact)

        if β < tol && K < howmany && verbosity >= WARN_LEVEL
            msg = "Invariant subspace of dimension $(K) (up to requested tolerance `tol = $tol`), "
            msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
            @warn msg
        end
        # BlockLanczos can access the case of K = 1 and doesn't need extra processing
        if K >= krylovdim || β <= tol || (alg.eager && K >= howmany)
            # compute eigenvalues
            # Note: Fast eigen solver for block tridiagonal matrices is not implemented yet.
            TDB = view(fact.TDB, 1:K, 1:K)
            D, U = eigen(Hermitian(TDB))
            by, rev = eigsort(which)
            p = sortperm(D; by=by, rev=rev)
            D, U = permuteeig!(D, U, p)

            # detect convergence by computing the residuals
            bs_r = fact.r_size   # the block size of the residual (decreases as the iteration goes)
            r = residual(fact)
            UU = U[(end - bs_r + 1):end, :]  # the last bs_r rows of U, used to compute the residuals
            normresiduals = diag(UU' * block_inner(r, r) * UU)
            normresiduals = sqrt.(real.(normresiduals))
            converged = count(nr -> nr <= tol, normresiduals)
            if converged >= howmany || β <= tol  # successfully find enough eigenvalues
                break
            elseif verbosity >= EACHITERATION_LEVEL
                @info "BlockLanczos eigsolve in iteration $numiter: $converged values converged, normres = $(normres2string(normresiduals))"
            end
        end

        if K < krylovdim
            expand!(iter, fact; verbosity=verbosity)
            numops += 1
        else # shrink and restart
            numiter >= maxiter && break
            bsn = max(div(3 * krylovdim + 2 * converged, 5) ÷ bs, 1) # Divide basis into blocks with the same size
            keep = bs * bsn
            H = zeros(S, (bsn + 1) * bs, bsn * bs)
            # The last bs rows of U contribute to calculate errors of Ritz values.
            @inbounds for j in 1:keep
                H[j, j] = D[j]
                H[(bsn * bs + 1):end, j] = U[(K - bs + 1):K, j]
            end
            # Turn diagonal matrix D into a block tridiagonal matrix, and make sure 
            # the residual of krylov subspace keeps the form of [0,..,0,R]
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + bs, 1:j, j)
                H[j + bs, j] = ν
                H[j + bs, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:(j + bs - 1), :), h')
                rmul!(U, h')
            end
            # transform the basis and update the residual and update the TDB.
            TDB .= S(0)
            TDB[1:keep, 1:keep] .= H[1:keep, 1:keep]
            B = basis(fact)
            basistransform!(B, view(U, :, 1:keep))

            r_new = OrthonormalBasis(fact.r.vec[1:bs_r])
            view_U = view(U, (K - bs_r + 1):K, (keep - bs_r + 1):keep)
            basistransform!(r_new, view_U)
            fact.r.vec[1:bs_r] = r_new[1:bs_r]

            while length(fact) > keep
                pop!(fact.V)
                fact.total_size -= 1
            end
            numiter += 1
        end
    end

    howmany_actual = howmany
    if converged > howmany
        howmany_actual = converged
    elseif length(D) < howmany
        howmany_actual = length(D)
    end
    U1 = view(U, :, 1:howmany_actual)
    vectors = let V = basis(fact)
        [V * u for u in cols(U1)]
    end
    bs_r = fact.r_size
    K = length(fact)
    U2 = view(U, (K - bs_r + 1):K, 1:howmany_actual)
    R = fact.r
    residuals = [zerovector(x₀) for _ in 1:howmany_actual]
    @inbounds for i in 1:howmany_actual
        for j in 1:bs_r
            residuals[i] = add!!(residuals[i], R[j], U2[j, i])
        end
    end
    normresiduals = normresiduals[1:howmany_actual]

    if (converged < howmany) && verbosity >= WARN_LEVEL
        @warn """BlockLanczos eigsolve stopped without full convergence after $(K) iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif verbosity >= STARTSTOP_LEVEL
        @info """BlockLanczos eigsolve finished after $(K) iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return D[1:howmany_actual],
           vectors[1:howmany_actual],
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
