function eigsolve(A, x₀::Block{T}, howmany::Int, which::Selector, alg::BlockLanczos;
                  alg_rrule=Arnoldi(; tol=alg.tol,
                                    krylovdim=alg.krylovdim,
                                    maxiter=alg.maxiter,
                                    eager=alg.eager,
                                    orth=alg.orth)) where {T}
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    if howmany > krylovdim
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end
    tol = alg.tol
    verbosity = alg.verbosity

    bs = length(x₀)
    iter = BlockLanczosIterator(A, x₀, krylovdim + bs, alg.orth, alg.qr_tol)
    fact = initialize(iter; verbosity=verbosity)  # Returns a BlockLanczosFactorization
    numops = bs + 1    # Number of matrix-vector multiplications (for logging)
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
            # Compute eigenvalues
            # Note: Fast eigen solver for block tridiagonal matrices is not implemented yet.
            BTD = view(fact.H, 1:K, 1:K)
            D, U = eigen(Hermitian(BTD))
            by, rev = eigsort(which)
            p = sortperm(D; by=by, rev=rev)
            D, U = permuteeig!(D, U, p)

            # Detect convergence by computing the residuals
            bs_R = fact.R_size   # The block size of the residual (decreases as the iteration goes)
            r = residual(fact)
            UU = view(U, (K - bs_R + 1):K, :)  # The last bs_R rows of U, used to compute the residuals
            normresiduals = let R = block_inner(r, r)
                map(u -> sqrt(real(dot(u, R, u))), cols(UU))
            end
            converged = 0
            while converged < K && normresiduals[converged + 1] <= tol
                converged += 1
            end
            if converged >= howmany || β <= tol  # Successfully find enough eigenvalues.
                break
            elseif verbosity >= EACHITERATION_LEVEL
                @info "BlockLanczos eigsolve in iteration $numiter, Krylov dimension = $K: $converged values converged, normres = $(normres2string(normresiduals[1:howmany]))"
            end
        end

        if K < krylovdim
            expand!(iter, fact; verbosity=verbosity)
            numops += fact.R_size
        else # Shrink and restart following the shrinking method of `Lanczos`.
            numiter >= maxiter && break
            keep = max(div(3 * krylovdim + 2 * converged, 5 * bs), 1) * bs
            H = zeros(eltype(fact.H), keep + bs, keep)
            # The last bs rows of U contribute to calculate errors of Ritz values.
            @inbounds for j in 1:keep
                H[j, j] = D[j]
                H[(keep + 1):end, j] = view(U, (K - bs + 1):K, j)
            end
            # Turn diagonal matrix D into a block tridiagonal matrix, and make sure 
            # The residual of krylov subspace keeps the form of [0,..,0,R]
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + bs, 1:j, j)
                H[j + bs, j] = ν
                H[j + bs, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:(j + bs - 1), :), h')
                rmul!(U, h')
            end
            # Transform the basis and update the residual and update the BTD.
            fill!(BTD, zero(eltype(BTD)))
            Hkeep = view(H, 1:keep, 1:keep)
            BTD[1:keep, 1:keep] .= (Hkeep .+ Hkeep') ./ 2 # make exactly Hermitian
            B = basis(fact)
            basistransform!(B, view(U, :, 1:keep))

            R_new = OrthonormalBasis(fact.R.vec[1:bs_R])
            view_H = view(H, (keep + bs - bs_R + 1):(keep + bs), (keep - bs_R + 1):keep)
            basistransform!(R_new, view_H)
            fact.R.vec[1:bs_R] = R_new[1:bs_R]

            while length(fact) > keep
                pop!(fact.V)
                fact.k -= 1
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
    values = D[1:howmany_actual]
    U1 = view(U, :, 1:howmany_actual)
    vectors = let V = basis(fact)
        [V * u for u in cols(U1)]
    end
    bs_R = fact.R_size
    K = length(fact)
    U2 = view(U, (K - bs_R + 1):K, 1:howmany_actual)
    R = fact.R
    residuals = [zerovector(R[1]) for _ in 1:howmany_actual]
    @inbounds for i in 1:howmany_actual
        for j in 1:bs_R
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

    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
