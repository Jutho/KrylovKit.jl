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

function eigsolve(A, x₀, howmany::Int, which::Selector, alg::BlockLanczos)
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    if howmany > krylovdim
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end
    tol = alg.tol
    verbosity = alg.verbosity
    x₀_vec = [randn!(similar(x₀)) for _ in 1:alg.blocksize-1]
    pushfirst!(x₀_vec, x₀)
    bs = length(x₀_vec)

    iter = BlockLanczosIterator(A, x₀_vec, krylovdim + bs, alg.qr_tol, alg.orth)
    fact = initialize(iter; verbosity = verbosity)
    numops = 2 # how many times we apply A
    numiter = 1
    vectors = [similar(x₀_vec[1]) for _ in 1:howmany]
    values = Vector{real(eltype(fact.TDB))}(undef, howmany)
    converged = false
    num_converged = 0
    local howmany_actual, residuals, normresiduals, D, U

    while true
        expand!(iter, fact; verbosity = verbosity)
        numops += 1

        # When norm(Rk) is to small, we may lose too much precision and orthogonalization.
        if fact.all_size > krylovdim || (fact.norm_r < tol) || (fact.r_size < 2)
            if fact.norm_r < tol && fact.all_size < howmany && verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $(fact.all_size) (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end

            all_size = fact.all_size
            TDB = view(fact.TDB, 1:all_size, 1:all_size)
            D, U = eigen(Hermitian((TDB + TDB') / 2))
            by, rev = eigsort(which)
            p = sortperm(D; by = by, rev = rev)
            D = D[p]
            U = U[:, p]
            T = eltype(fact.V.basis)
            S = eltype(TDB)
        
            howmany_actual = min(howmany, length(D))
            copyto!(values, D[1:howmany_actual])
        
            residuals = Vector{T}(undef, howmany_actual)
            normresiduals = Vector{real(S)}(undef, howmany_actual)
            bs_r = fact.r_size
            r = fact.r[1:bs_r]
            UU = U[end-bs_r+1:end, :]
            for i in 1:howmany_actual
                residuals[i] = r[1] * UU[1, i]
                for j in 2:bs_r
                    axpy!(UU[j, i], r[j], residuals[i])
                end
                normresiduals[i] = norm(residuals[i])
            end
            num_converged = count(nr -> nr <= tol, normresiduals)

            if num_converged >= howmany || fact.norm_r < tol
                converged = true
                break
            elseif verbosity >= EACHITERATION_LEVEL
                @info "Block Lanczos eigsolve in iteration $numiter: $num_converged values converged, normres = $(normres2string(normresiduals[1:howmany]))"
            end
            if fact.all_size > krylovdim # begin to shrink dimension
                numiter >= maxiter && break
                bsn = max(div(3 * krylovdim + 2 * num_converged, 5) ÷ bs, 1)
                if (bsn + 1) * bs > fact.all_size # make sure that we can fetch next block after shrinked dimension as residual
                    warning("shrinked dimesion is too small and there is no need to shrink")
                    break
                end
                keep = bs * bsn
                H = zeros(S, (bsn + 1) * bs, bsn * bs)
                @inbounds for j in 1:keep
                    H[j, j] = D[j]
                    H[bsn * bs + 1:end, j] = U[all_size - bs + 1:all_size, j]
                end
                @inbounds for j in keep:-1:1
                    h, ν = householder(H, j + bs, 1:j, j)
                    H[j + bs, j] = ν
                    H[j + bs, 1:(j - 1)] .= zero(eltype(H))
                    lmul!(h, H)
                    rmul!(view(H, 1:j + bs -1, :), h')
                    rmul!(U, h')
                end
                TDB .= S(0)
                TDB[1:keep, 1:keep] .= H[1:keep, 1:keep]

                V = OrthonormalBasis(fact.V.basis[1:all_size])
                basistransform!(V, view(U, :, 1:keep))
                fact.V[1:keep] = V[1:keep]

                r_new = OrthonormalBasis(fact.r.vec[1:bs_r])
                view_U = view(U, all_size - bs_r + 1:all_size, keep - bs_r + 1:keep)
                basistransform!(r_new, view_U)
                fact.r.vec[1:bs_r] = r_new[1:bs_r]

                fact.all_size = keep
                numiter += 1
            end
        end
    end
    V = view(fact.V.basis, 1:fact.all_size)
    @inbounds for i in 1:howmany_actual
        copy!(vectors[i], V[1] * U[1, i])
        for j in 2:fact.all_size
            axpy!(U[j, i], V[j], vectors[i])
        end
    end

    if (num_converged < howmany) && verbosity >= WARN_LEVEL
        @warn """Block Lanczos eigsolve stopped without full convergence after $(fact.all_size) iterations:
        * $num_converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif verbosity >= STARTSTOP_LEVEL
        @info """Block Lanczos eigsolve finished after $(fact.all_size) iterations:
        * $num_converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values[1:howmany_actual],
    vectors[1:howmany_actual],
    ConvergenceInfo(num_converged, residuals, normresiduals, numiter, numops)
end