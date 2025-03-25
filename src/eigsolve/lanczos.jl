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


function eigsolve(A, x₀, howmany::Int, which::Selector, alg::BlockLanczos;
                  tol = 1e-10, maxiter = 200)
    ## FIRST ITERATION: setting up
    # Initialize block size and check dimensions
    p = alg.block_size
    n = size(A, 1)
    if n % p != 0
        error("Matrix dimension $n must be divisible by block size $p")
    end

    # Initialize first block and allocate storage
    X = [x₀]
    M₁ = x₀' * A * x₀
    M = [(M₁ + M₁') / 2]
    AX = A * x₀
    R = AX - x₀ * M[1]
    X₂, B = qr(R)
    X₂ = Matrix(X₂)
    X₂ = X₂ - x₀ * (x₀' * X₂)
    X₂ = X₂ ./ sqrt.(sum(abs2.(X₂), dims = 1))
    B = [X₂' * R]
    M₂ = X₂' * A * X₂
    push!(X, X₂)
    push!(M, (M₂ + M₂') / 2)

    # Initialize iteration counters
    numiter = 1
    numops = 4
    r = ceil(Int64, n / p)

    ## MAIN ITERATION
    while true
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn "Block Lanczos eigsolve reached maximum number of iterations ($maxiter)"
            end
            break
        end

        # Expand Krylov subspace
        k = length(X)
        R = A * X[k] - X[k] * M[k] - X[k-1] * B[k-1]'
        Xnext, Bcurr = qr(R)
        Xnext = Matrix(Xnext)

        # Orthogonalize against previous vectors
        for Y in X
            Xnext = Xnext - Y * (Y' * Xnext)
        end
        Xnext = Xnext ./ sqrt.(sum(abs2.(Xnext), dims = 1))
        Bcurr = Xnext' * R

        # Update matrices
        push!(X, Xnext)
        push!(B, Bcurr)
        Mnext = Xnext' * A * Xnext
        Mnext = (Mnext + Mnext') / 2
        push!(M, Mnext)
        numops += 2

        # Check convergence
        if norm(Bcurr) < tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info "Block Lanczos converged after $numiter iterations with residual $(norm(Bcurr))"
            end
            break
        end

        if k >= r - 1
            if alg.verbosity >= STARTSTOP_LEVEL
                @info "Block Lanczos reached maximum subspace dimension after $numiter iterations"
            end
            break
        end

        if alg.verbosity >= EACHITERATION_LEVEL
            @info "Block Lanczos iteration $numiter: residual = $(norm(Bcurr))"
        end

        numiter += 1
    end

    # Construct and diagonalize block tridiagonal matrix
    m = length(M)
    TDB = zeros(m * p, m * p)
    for i in 1:m
        TDB[i*p-p+1:i*p, i*p-p+1:i*p] = M[i]
        if i != m
            TDB[i*p-p+1:i*p, i*p+1:(i+1)*p] = B[i]'
            TDB[i*p+1:(i+1)*p, i*p-p+1:i*p] = B[i]
        end
    end

    # Compute eigenvalues and eigenvectors
    D, U = LinearAlgebra.eigen(TDB)
    by, rev = eigsort(which)
    p = sortperm(D; by = by, rev = rev)
    D, U = permuteeig!(D, U, p)
    
    # Select requested number of eigenvalues/vectors
    howmany′ = min(howmany, length(D))
    values = D[1:howmany′]
    vectors = hcat(X...) * U[:, 1:howmany′]

    # Compute convergence information
    residuals = [A * v - λ * v for (v, λ) in zip(eachcol(vectors), values)]
    normresiduals = [norm(r) for r in residuals]
    converged = count(x -> x < tol, normresiduals)

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """Block Lanczos eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """Block Lanczos eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values, vectors, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
