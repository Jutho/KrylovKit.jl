# Arnoldi methods for eigenvalue problems
function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{NoRestart})
    krylovdim = min(alg.krylovdim, length(x₀))
    howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    # Compute arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
    fact = start(iter)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    numops = 1
    while length(fact) < krylovdim
        fact = next!(iter, fact)
        numops += 1
        normres(fact) < tol && break
    end

    # Process
    # Dense Schur factorization
    β = normres(fact)
    H = copy(matrix(fact)) # creates a Matrix copy
    T, U, values = hschur!(H, one(H))
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    T, U = permuteschur!(T, U, p)
    f = scale!(map(last, cols(U)), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end

    # Compute eigenvectors
    if eltype(H) <: Real && fact.k > howmany && H[howmany+1,howmany] != 0
        howmany += 1
    end
    values = schur2eigvals(H)[1:howmany]
    R = schur2eigvecs(H, 1:howmany)
    V = U*R;

    # Compute convergence information
    vectors = let B = basis(fact)
        [B*v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [r*last(v) for v in cols(V)]
    end
    normreseigvecs = β*map(v->abs(last(v)), cols(V))

    return values, vectors, ConvergenceInfo(converged, normreseigvecs, residuals, 1, numops)
end

function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{ExplicitRestart})
    howmany == 1 || error("explicit restart currently only implemented for computing single eigenvalue")
    krylovdim = min(alg.krylovdim, length(x₀))
    maxiter = alg.restart.maxiter
    howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
    fact = start(iter)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    numops = 1
    while length(fact) < krylovdim
        fact = next!(iter, fact)
        numops += 1
        normres(fact) < tol && break
    end

    # Process
    # allocate storage
    HH = zeros(eltype(fact), krylovdim+1, krylovdim)
    UU = zeros(eltype(fact), krylovdim, krylovdim)

    # initialize
    β = normres(fact)
    m = length(fact)
    H = view(HH, 1:m, 1:m)
    U = view(UU, 1:m, 1:m)
    f = view(HH, m+1, 1:m)
    copy!(U, I)
    copy!(H, matrix(fact))

    # compute dense schur factorization
    T, U, values = hschur!(H, U)
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    T, U = permuteschur!(T, U, p)
    scale!(f, view(U,m,:), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        x₀ = basis(fact)*view(U, 1:m, 1) # use first Schur vector as new starting vector
        numiter += 1
        # Arnoldi factorization: recylce fact
        iter = ArnoldiIterator(A, x₀, alg.orth)
        fact = start!(iter, fact)
        numops += 1
        while length(fact) < krylovdim
            fact = next!(iter, fact)
            numops += 1
            normres(fact) < tol && break
        end

        # post process
        β = normres(fact)
        m = length(fact)
        H = view(HH, 1:m, 1:m)
        U = view(UU, 1:m, 1:m)
        f = view(HH, m+1, 1:m)
        copy!(U, I)
        copy!(H, matrix(fact))

        # compute dense schur factorization
        T, U, values = hschur!(H, U)
        by, rev = eigsort(which)
        p = sortperm(values, by = by, rev = rev)
        T, U = permuteschur!(T, U, p)
        scale!(f, view(U,m,:), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end
    end

    # Compute eigenvectors
    if eltype(H) <: Real && length(fact) > howmany && T[howmany+1,howmany] != 0
        howmany += 1
    end
    values = schur2eigvals(H, 1:howmany)
    R = schur2eigvecs(H, 1:howmany)
    V = U*R;

    # Compute convergence information
    vectors = let B = basis(fact)
        [B*v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [r*last(v) for v in cols(V)]
    end
    normreseigvecs = scale!(map(abs, view(V, m, :)), β)

    return values, vectors, ConvergenceInfo(converged, normreseigvecs, residuals, numiter, numops)
end

function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{ImplicitRestart})
    krylovdim = min(alg.krylovdim, length(x₀))
    maxiter = alg.restart.maxiter
    howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
    fact = start(iter)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    numops = 1
    while length(fact) < krylovdim
        fact = next!(iter, fact)
        numops += 1
        normres(fact) < tol && break
    end

    # Process
    # allocate storage
    HH = zeros(eltype(fact), krylovdim+1, krylovdim)
    UU = zeros(eltype(fact), krylovdim, krylovdim)

    # initialize
    β = normres(fact)
    m = length(fact)
    H = view(HH, 1:m, 1:m)
    U = view(UU, 1:m, 1:m)
    f = view(HH, m+1, 1:m)
    copy!(U, I)
    copy!(H, matrix(fact))

    # compute dense schur factorization
    T, U, values = hschur!(H, U)
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    T, U = permuteschur!(T, U, p)
    scale!(f, view(U,m,:), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        numiter += 1

        # Determine how many to keep
        keep = converged + ((krylovdim-converged)>>2) # strictly smaller than krylovdim
        if eltype(H) <: Real && H[keep+1,keep] != 0 # we are in the middle of a 2x2 block
            keep += 1 # conservative choice
            keep >= krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
        end

        # Update B by applying U using Householder reflections
        B = basis(fact)
        for j = 1:m
            h, ν = householder(U, j:m, j)
            lmul!(U, h, j+1:krylovdim)
            rmulc!(B, h)
        end

        # Shrink Krylov factorization (no longer strictly Krylov)
        B[keep+1] = last(B)
        for j = converged+1:keep
            H[keep+1,j] = f[j]
        end

        # Restore Krylov form in the first keep columns
        for j = keep:-1:1
            h, ν = householder(H, j+1, 1:j, j)
            H[j+1,j] = ν
            @inbounds H[j+1,1:j-1] = 0
            lmul!(H, h)
            rmulc!(H, h, 1:j)
            rmulc!(B, h)
        end
        copy!(matrix(fact), H) # copy back into fact
        fact = shrink!(fact, keep)

        # Arnoldi factorization: recylce fact
        while length(fact) < krylovdim
            fact = next!(iter, fact)
            numops += 1
            normres(fact) < tol && break
        end

        # post process
        β = normres(fact)
        m = length(fact)
        H = view(HH, 1:m, 1:m)
        U = view(UU, 1:m, 1:m)
        f = view(HH, m+1, 1:m)
        copy!(U, I)
        copy!(H, matrix(fact))

        # compute dense schur factorization
        T, U, values = hschur!(H, U)
        by, rev = eigsort(which)
        p = sortperm(values, by = by, rev = rev)
        T, U = permuteschur!(T, U, p)
        scale!(f, view(U,m,:), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end
    end
    # Compute eigenvectors
    if eltype(H) <: Real && length(fact) > howmany && T[howmany+1,howmany] != 0
        howmany += 1
    end
    values = schur2eigvals(T, 1:howmany)
    R = schur2eigvecs(T, 1:howmany)
    V = U*R;

    # Compute convergence information
    vectors = let B = basis(fact)
        [B*v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [r*last(v) for v in cols(V)]
    end
    normreseigvecs = scale!(map(abs, view(V, m, :)), β)

    return values, vectors, ConvergenceInfo(converged, normreseigvecs, residuals, numiter, numops)
end
