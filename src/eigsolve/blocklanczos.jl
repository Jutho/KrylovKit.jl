function eigsolve(
        A, x₀::AbstractBlock, howmany::Int, which::Selector, alg::BlockLanczos;
        alg_rrule = Arnoldi(;
            tol = alg.tol,
            krylovdim = alg.krylovdim,
            maxiter = alg.maxiter,
            eager = alg.eager,
            orth = alg.orth
        )
    )
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    if howmany > krylovdim
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end
    tol = alg.tol
    verbosity = alg.verbosity

    bs = length(x₀)
    iter = BlockLanczosIterator(A, x₀, krylovdim + bs, alg.orth, alg.qr_tol)
    fact = initialize(iter; verbosity = verbosity)  # Returns a BlockLanczosFactorization
    numops = bs + (x₀ isa Block) # Number of matrix-vector multiplications (for logging); a Block costs one extra in `initialize`
    numiter = 1

    converged = 0
    local normresiduals, D, U

    while true
        K = length(fact)
        β = normres(fact)

        if β < tol && K < howmany && verbosity >= WARN_LEVEL
            _warn_invariant_subspace(K, tol, howmany)
        end
        # BlockLanczos can access the case of K = 1 and doesn't need extra processing
        if K >= krylovdim || β <= tol || (alg.eager && K >= howmany)
            # Compute eigenvalues
            # Note: Fast eigen solver for block tridiagonal matrices is not implemented yet.
            BTD = view(fact.H, 1:K, 1:K)
            D, U = eigen(Hermitian(BTD))
            by, rev = eigsort(which)
            p = sortperm(D; by = by, rev = rev)
            D, U = permuteeig!(D, U, p)

            # Detect convergence by computing the residuals
            bs_R = fact.R_size   # The block size of the residual (decreases as the iteration goes)
            r = residual(fact)
            UU = view(U, (K - bs_R + 1):K, :)  # The last bs_R rows of U, used to compute the residuals
            normresiduals = let R = block_inner(r, r)
                map(u -> sqrt(max(real(dot(u, R, u)), 0)), cols(UU)) # (round-off can make it negative)
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
            expand!(iter, fact; verbosity = verbosity)
            numops += fact.R_size
        else # Shrink and restart following the shrinking method of `Lanczos`.
            numiter >= maxiter && break
            keep = max(div(3 * krylovdim + 2 * converged, 5 * bs_R), 1) * bs_R
            # Rounding down to a multiple of the block size can drop converged Ritz vectors;
            # then use the unrounded `Lanczos` value, which satisfies `converged <= keep < krylovdim`.
            keep < converged && (keep = div(3 * krylovdim + 2 * converged, 5))
            H = zeros(eltype(fact.H), keep + bs_R, keep)
            # The last bs_R rows of U contribute to calculate errors of Ritz values.
            @inbounds for j in 1:keep
                H[j, j] = D[j]
                H[(keep + 1):end, j] = view(U, (K - bs_R + 1):K, j)
            end
            # Turn diagonal matrix D into a block tridiagonal matrix, and make sure
            # The residual of krylov subspace keeps the form of [0,..,0,R]
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + bs_R, 1:j, j)
                H[j + bs_R, j] = ν
                H[j + bs_R, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:(j + bs_R - 1), :), h')
                rmul!(U, h')
            end
            # Transform the basis and update the residual and update the BTD.
            fill!(BTD, zero(eltype(BTD)))
            Hkeep = view(H, 1:keep, 1:keep)
            BTD[1:keep, 1:keep] .= (Hkeep .+ Hkeep') ./ 2 # make exactly Hermitian
            view_H = view(H, (keep + 1):(keep + bs_R), (keep - bs_R + 1):keep)
            shrink_basis!(fact, view(U, :, 1:keep), view_H)
            numiter += 1
        end
    end

    howmany′ = max(converged, min(howmany, length(D)))
    values = D[1:howmany′]
    U1 = view(U, :, 1:howmany′)
    vectors = ritzvectors(basis(fact), U1)
    bs_R = fact.R_size
    K = length(fact)
    U2 = view(U, (K - bs_R + 1):K, 1:howmany′)
    residuals = ritzresiduals(fact.R, U2, bs_R)
    normresiduals = normresiduals[1:howmany′]

    if (converged < howmany′) && verbosity >= WARN_LEVEL # all values are converged after an invariant subspace
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

    return values, vectors,
        ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

# the basis `V U` and residual `R H` of a thick restart that keeps `size(U, 2)` vectors
function shrink_basis!(fact::BlockLanczosFactorization, U, H)
    bs_R = size(H, 1)
    basistransform!(basis(fact), U)
    R_new = OrthonormalBasis(fact.R.vec[1:bs_R])
    basistransform!(R_new, H)
    fact.R.vec[1:bs_R] = R_new[1:bs_R]
    while length(fact) > size(U, 2)
        pop!(fact.V)
        fact.k -= 1
    end
    return fact
end
# a MatrixBlock basis becomes two blocks, the last one of the residual's width as the next expansion needs
function shrink_basis!(fact::MatrixBlockLanczosFactorization, U, H)
    keep = size(U, 2)
    bs_R = size(H, 1)
    V = fact.V
    blocks = [basis_mul(V, view(U, :, (keep - bs_R + 1):keep))]
    keep > bs_R && pushfirst!(blocks, basis_mul(V, view(U, :, 1:(keep - bs_R))))
    copy!(V.blocks, blocks)
    fact.R = block_mul(fact.R, H)
    fact.k = keep
    return fact
end

ritzvectors(V::OrthonormalBasis, U) = [V * u for u in cols(U)]
ritzvectors(V::MatrixBlockBasis, U) = basis_mul(V, U)

function ritzresiduals(R::Block, U, bs_R)
    residuals = [zerovector(R[1]) for _ in 1:size(U, 2)]
    @inbounds for i in 1:size(U, 2)
        for j in 1:bs_R
            residuals[i] = add!!(residuals[i], R[j], U[j, i])
        end
    end
    return residuals
end
ritzresiduals(R::MatrixBlock, U, bs_R) = block_mul(R, U)
