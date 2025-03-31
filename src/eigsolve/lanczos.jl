function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos;
                  alg_rrule=Arnoldi(; tol=alg.tol,
                                    krylovdim=alg.krylovdim,
                                    maxiter=alg.maxiter,
                                    eager=alg.eager,
                                    orth=alg.orth))
    if alg.block_size > 1
        return block_lanczos_reortho(A, x₀, howmany, which, alg)
    end
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

function block_lanczos_reortho(A, x₀, howmany::Int, which::Selector,
                               alg::Lanczos) where {S}
    @assert alg.block_size == blocksize(x₀)
    block_size = alg.block_size
    maxiter = alg.maxiter
    tol = alg.tol
    verbosity = alg.verbosity
    n = size(x₀, 1)
    max_blocks = n ÷ block_size

    iter = BlockLanczosIterator(A, x₀, block_size, maxiter, alg.orth)
    fact = initialize(iter; verbosity=verbosity)
    numops = 2 # how many times we apply A

    converge_check = max(1, 100 ÷ block_size) # Periodic check for convergence

    local values, vectors, residuals, normresiduals, num_converged
    converged = false

    for numiter in 2:min(maxiter, max_blocks - 2)
        expand!(iter, fact; verbosity=verbosity)
        numops += 1

        # Although norm(Rk) is not our convergence condition, when norm(Rk) is to small, we may lose too much precision and orthogonalization.
        if (numiter % converge_check == 0) || (fact.normR < tol)
            values, vectors, residuals, normresiduals, num_converged = _residual(fact, A,
                                                                            howmany, tol,
                                                                            block_size, which)

            if verbosity >= EACHITERATION_LEVEL
                @info "Block Lanczos eigsolve in iteration $numiter: $num_converged values converged, normres = $(normres2string(normresiduals[1:min(howmany, length(normresiduals))]))"
            end

            # This convergence condition refers to https://www.netlib.org/utk/people/JackDongarra/etemplates/node251.html
            if num_converged >= howmany || fact.normR < tol
                converged = true
                break
            end
        end
    end

    if !converged
        values, vectors, residuals, normresiduals, num_converged = _residual(fact, A, howmany,
                                                                            tol, block_size, which)
    end

    if (fact.k * block_size > alg.krylovdim)
        @warn "The real Krylov dimension is $(fact.k * block_size), which is larger than the maximum allowed dimension $(alg.krylovdim)."
        # In this version we don't shrink the factorization because it might cause issues, different from the ordinary Lanczos.
        # Why it happens remains to be investigated.
    end

    if (num_converged < howmany) && verbosity >= WARN_LEVEL
        @warn """Block Lanczos eigsolve stopped without full convergence after $(fact.k) iterations:
        * $num_converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif verbosity >= STARTSTOP_LEVEL
        @info """Block Lanczos eigsolve finished after $(fact.k) iterations:
        * $num_converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values,
           vectors,
           ConvergenceInfo(num_converged, residuals, normresiduals, fact.k, numops)
end

function _residual(fact, A, howmany, tol, block_size, which)
    T = triblockdiag(fact)
    D, U = eigen(Hermitian((T + T') / 2))   # TODO: use keyword sortby

    by, rev = eigsort(which)
    p = sortperm(D; by=by, rev=rev)
    D = D[p]
    U = U[:, p]

    howmany_actual = min(howmany, length(D))
    values = D[1:howmany_actual]

    basis_so_far = view(fact.V, :, 1:(fact.k * block_size))
    vectors = Vector{typeof(fact.V[:, 1])}(undef, howmany_actual)

    for i in 1:howmany_actual
        vectors[i] = similar(basis_so_far, size(basis_so_far, 1))
        mul!(vectors[i], basis_so_far, view(U, :, i))
    end

    residuals = Vector{typeof(vectors[1])}(undef, howmany_actual)
    normresiduals = Vector{Float64}(undef, howmany_actual)

    for i in 1:howmany_actual
        residuals[i] = apply(A, vectors[i])
        axpy!(-values[i], vectors[i], residuals[i])  # residuals[i] -= values[i] * vectors[i]
        normresiduals[i] = norm(residuals[i])
    end

    num_converged = count(nr -> nr <= tol, normresiduals)
    return values, vectors, residuals, normresiduals, num_converged
end