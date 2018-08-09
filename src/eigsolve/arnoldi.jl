# Arnoldi methods for eigenvalue problems
function schursolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
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
    H = view(HH, 1:m, 1:m)
    U = view(UU, 1:m, 1:m)
    f = view(HH, m+1, 1:m)
    copyto!(U, I)
    copyto!(H, rayleighquotient(fact))

    # compute dense schur factorization
    T, U, values = hschur!(H, U)
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    T, U = permuteschur!(T, U, p)
    mul!(f, view(U,m,:), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end
    if eltype(T) <: Real && 0< converged < length(fact) && T[converged+1,converged] != 0
        converged -= 1
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        numiter += 1

        # Determine how many to keep
        keep = div(3*krylovdim + 2*converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
        if eltype(H) <: Real && H[keep+1,keep] != 0 # we are in the middle of a 2x2 block
            keep += 1 # conservative choice
            keep >= krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
        end

        # Update B by applying U using Householder reflections
        B = basis(fact)
        for j = 1:m
            h, ν = householder(U, j:m, j)
            lmul!(h, view(U, :, j+1:krylovdim))
            rmul!(B, h')
        end

        # Shrink Arnoldi factorization (no longer strictly Arnoldi but still Krylov)
        r = residual(fact)
        B[keep+1] = mul!(r, r, 1/normres(fact))
        for j = 1:keep
            H[keep+1,j] = f[j]
        end

        # Restore Arnoldi form in the first keep columns
        for j = keep:-1:1
            h, ν = householder(H, j+1, 1:j, j)
            H[j+1,j] = ν
            @inbounds H[j+1,1:j-1] .= 0
            lmul!(h, H)
            rmul!(view(H, 1:j,:), h')
            rmul!(B, h')
        end
        copyto!(rayleighquotient(fact), H) # copy back into fact
        fact = shrink!(fact, keep)

        # Arnoldi factorization: recylce fact
        while length(fact) < krylovdim
            fact = expand!(iter, fact)
            numops += 1
            normres(fact) < tol && length(fact) >= howmany && break
        end

        # post process
        β = normres(fact)
        m = length(fact)
        H = view(HH, 1:m, 1:m)
        U = view(UU, 1:m, 1:m)
        f = view(HH, m+1, 1:m)
        copyto!(U, I)
        copyto!(H, rayleighquotient(fact))

        # compute dense schur factorization
        T, U, values = hschur!(H, U)
        by, rev = eigsort(which)
        p = sortperm(values, by = by, rev = rev)
        T, U = permuteschur!(T, U, p)
        mul!(f, view(U,m,:), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end
        if eltype(T) <: Real && 0 < converged < length(fact) && T[converged+1,converged] != 0
            converged -= 1
        end
    end
    if eltype(T) <: Real && howmany < length(fact) && T[howmany+1,howmany] != 0
        howmany += 1
    end
    if converged > howmany
        howmany = converged
    end
    TT = view(T,1:howmany,1:howmany)
    values = schur2eigvals(TT)
    vectors = let B = basis(fact)
        [B*u for u in cols(U, 1:howmany)]
    end
    residuals = let r = residual(fact)
        [r*last(u) for u in cols(U, 1:howmany)]
    end
    normresiduals = let f = f
        map(i->abs(f[i]), 1:howmany)
    end

    return view(T,1:howmany,1:howmany), vectors, values, ConvergenceInfo(converged, normresiduals, residuals, numiter, numops)
end

function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    T, schurvectors, values, info = schursolve(A, x₀, howmany, which, alg)

    # Transform schurvectors to eigenvectors
    values = schur2eigvals(T)
    V = schur2eigvecs(T)
    vectors = let B = OrthonormalBasis(schurvectors)
        [B*v for v in cols(V)]
    end

    return values, vectors, info
end
