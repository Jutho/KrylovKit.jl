function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute Lanczos factorization
    iter = LanczosIterator(A, x₀, alg.orth)
    fact = initialize(iter)
    numops = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    while length(fact) < krylovdim
        fact = expand!(iter, fact)
        numops += 1
        normres(fact) < tol && length(fact) >= howmany && break
    end

    # Process
    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim+1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize
    β = normres(fact)
    m = length(fact)
    U = copyto!(view(UU, 1:m, 1:m), I)
    f = view(HH, m+1, 1:m)
    T = rayleighquotient(fact) # symtridiagonal

    # compute eigenvalues
    D, U = eig!(T, U)
    by, rev = eigsort(which)
    p = sortperm(D, by = by, rev = rev)
    D, U = permuteeig!(D, U, p)
    mul!(f, view(U,m,:), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        numiter += 1

        # Determine how many to keep
        keep = div(3*krylovdim + 2*converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

        # Update B by applying U using Householder reflections
        B = basis(fact)
        basistransform!(B, view(U, :, 1:keep))
        # for j = 1:m
        #     h, ν = householder(U, j:m, j)
        #     lmul!(h, view(U, :, j+1:krylovdim))
        #     rmul!(B, h')
        # end

        # Shrink Lanczos factorization (no longer strictly Lanczos)
        r = residual(fact)
        B[keep+1] = rmul!(r, 1/normres(fact))
        H = fill!(view(HH, 1:keep+1, 1:keep), 0)
        @inbounds for j = 1:keep
            H[j,j] = D[j]
            H[keep+1,j] = f[j]
        end

        # Restore Lanczos form in the first keep columns
        for j = keep:-1:1
            h, ν = householder(H, j+1, 1:j, j)
            H[j+1,j] = ν
            @inbounds H[j+1,1:j-1] .= 0
            lmul!(h, H)
            rmul!(view(H, 1:j, :), h')
            rmul!(B, h')
        end
        @inbounds for j = 1:keep
            fact.αs[j] = H[j,j]
            fact.βs[j] = H[j+1,j]
        end
        fact = shrink!(fact, keep)

        # Lanczos factorization: recylce fact
        while length(fact) < krylovdim
            fact = expand!(iter, fact)
            numops += 1
            normres(fact) < tol && length(fact) >= howmany && break
        end

        # post process
        β = normres(fact)
        m = length(fact)
        U = copyto!(view(UU, 1:m, 1:m), I)
        f = view(HH, m+1, 1:m)
        T = rayleighquotient(fact) # symtridiagonal

        # compute eigenvalues
        D, U = eig!(T, U)
        by, rev = eigsort(which)
        p = sortperm(D, by = by, rev = rev)
        D, U = permuteeig!(D, U, p)
        mul!(f, view(U,m,:), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end
    end

    if converged > howmany
        howmany = converged
    end
    values = D[1:howmany]

    # Compute eigenvectors
    V = view(U,:,1:howmany)

    # Compute convergence information
    vectors = let B = basis(fact)
        [B*v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [r*last(v) for v in cols(V)]
    end
    normresiduals = let f = f
        map(i->abs(f[i]), 1:howmany)
    end

    return values, vectors, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
