function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute Lanczos factorization
    iter = LanczosIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity = alg.verbosity - 2)
    numops = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    if normres(fact) > tol || howmany > 1
        while length(fact) < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
            numops += 1
            normres(fact) <= tol && length(fact) >= howmany && break
        end
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
    D, U = tridiageigh!(T, U)
    by, rev = eigsort(which)
    p = sortperm(D, by = by, rev = rev)
    D, U = permuteeig!(D, U, p)
    mul!(f, view(U, m, :), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end

    if alg.verbosity > 1
        msg = "Lanczos eigsolve in iter $numiter: "
        msg *= "$converged values converged, normres = ("
        msg *= @sprintf("%.2e", abs(f[1]))
        for i = 2:howmany
            msg *= ", "
            msg *= @sprintf("%.2e", abs(f[i]))
        end
        msg *= ")"
        @info msg
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        numiter += 1

        # Determine how many to keep
        keep = div(3*krylovdim + 2*converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

        # Restore Lanczos form in the first keep columns
        H = fill!(view(HH, 1:keep+1, 1:keep), 0)
        @inbounds for j = 1:keep
            H[j,j] = D[j]
            H[keep+1,j] = f[j]
        end
        @inbounds for j = keep:-1:1
            h, ν = householder(H, j+1, 1:j, j)
            H[j+1,j] = ν
            H[j+1,1:j-1] .= 0
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
        # for j = 1:m
        #     h, ν = householder(U, j:m, j)
        #     lmul!(h, view(U, :, j+1:krylovdim))
        #     rmul!(B, h')
        # end
        r = residual(fact)
        B[keep+1] = rmul!(r, 1/β)

        # Shrink Lanczos factorization
        fact = shrink!(fact, keep)

        # Lanczos factorization: recylce fact
        while length(fact) < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
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
        D, U = tridiageigh!(T, U)
        by, rev = eigsort(which)
        p = sortperm(D, by = by, rev = rev)
        D, U = permuteeig!(D, U, p)
        mul!(f, view(U,m,:), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end

        if alg.verbosity > 1
            msg = "Lanczos eigsolve in iter $numiter: "
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
        [r*last(v) for v in cols(V)]
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
