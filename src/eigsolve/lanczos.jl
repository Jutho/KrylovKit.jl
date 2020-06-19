function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up

    # Initialize Lanczos factorization
    iter = LanczosIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity = alg.verbosity - 2)
    numops = 1
    numiter = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::typeof(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim+1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    converged = 0
    local D, U, f
    while true
        β = normres(fact)
        K = length(fact)

        # diagonalize Krylov factorization
        if β <= tol
            if K < howmany
                @warn "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`); setting `howmany = $K`."
                howmany = K
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && K >= howmany)
            U = copyto!(view(UU, 1:K, 1:K), I)
            f = view(HH, K+1, 1:K)
            T = rayleighquotient(fact) # symtridiagonal

            # compute eigenvalues
            if K == 1
                D = [T[1,1]]
                f[1] = β
                converged = Int(β <= tol)
            else
                if K < krylovdim
                    T = deepcopy(T)
                end
                D, U = tridiageigh!(T, U)
                by, rev = eigsort(which)
                p = sortperm(D, by = by, rev = rev)
                D, U = permuteeig!(D, U, p)
                mul!(f, view(U, K, :), β)
                converged = 0
                while converged < K && abs(f[converged+1]) <= tol
                    converged += 1
                end
            end

            if converged >= howmany
                break
            elseif alg.verbosity > 1
                msg = "Lanczos eigsolve in iter $numiter, krylovdim = $K: "
                msg *= "$converged values converged, normres = ("
                msg *= @sprintf("%.2e", abs(f[1]))
                for i = 2:howmany
                    msg *= ", "
                    msg *= @sprintf("%.2e", abs(f[i]))
                end
                msg *= ")"
                @info msg
            end
        end

        if K < krylovdim# expand Krylov factorization
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
            numops += 1
        else ## shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many to keep
            keep = div(3*krylovdim + 2*converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            # Restore Lanczos form in the first keep columns
            H = fill!(view(HH, 1:keep+1, 1:keep), zero(eltype(HH)))
            @inbounds for j = 1:keep
                H[j,j] = D[j]
                H[keep+1,j] = f[j]
            end
            @inbounds for j = keep:-1:1
                h, ν = householder(H, j+1, 1:j, j)
                H[j+1,j] = ν
                H[j+1,1:j-1] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:j, :), h')
                rmul!(U, h')
            end
            @inbounds for j = 1:keep
                fact.αs[j] = H[j,j]
                fact.βs[j] = H[j+1,j]
            end

            # Update B by applying U using Householder reflections
            B = basis(fact)
            basistransform!(B, view(U, :, 1:keep))
            r = residual(fact)
            B[keep+1] = rmul!(r, 1/β)

            # Shrink Lanczos factorization
            fact = shrink!(fact, keep)
            numiter += 1
        end
    end

    if converged > howmany
        howmany = converged
    end
    values = D[1:howmany]

    # Compute eigenvectors
    V = view(U, :, 1:howmany)

    # Compute convergence information
    vectors = let B = basis(fact)
        [B*v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [last(v)*r for v in cols(V)]
    end
    normresiduals = let f = f
        map(i->abs(f[i]), 1:howmany)
    end

    if alg.verbosity > 0
        if converged < howmany
            @warn """Lanczos eigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Lanczos eigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end

    return values, vectors, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
