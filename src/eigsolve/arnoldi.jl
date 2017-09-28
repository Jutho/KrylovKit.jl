# Arnoldi methods for eigenvalue problems
function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{NoRestart})
    krylovdim = min(alg.krylovdim, length(x₀))
    if eltype(x₀) <: Real
        howmany < krylovdim-1 || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    else
        howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end

    # Compute arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth; krylovdim = krylovdim, tol = alg.tol)
    fact = start(iter)
    numops = 0
    while !done(iter, fact) || fact.k < howmany
        next!(iter, fact)
        numops += 1
    end

    # Post-process
    H = view(fact.H, 1:fact.k, 1:fact.k)
    T, U, values = schur!(H, one(H))
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    reorderschur!(T, p, U)
    if eltype(H) <: Real && fact.k > howmany && H[howmany+1,howmany] != 0
        howmany += 1
    end
    values = values[p[1:howmany]]
    # R = schur2rightvecs(H, 1:howmany)
    # V = U*R;
    R = schur2rightvecs(H, 1:fact.k)
    V = U*R;

    # Compute convergence information
    r = residual(fact)
    β = normres(fact)
    vectors = Vector{typeof(r)}(howmany)
    residuals = Vector{typeof(r)}(howmany)
    nres = Vector{typeof(β)}(howmany)
    converged = 0
    @inbounds for i = 1:howmany
        vectors[i] = fact.V*view(V,:,i)
        residuals[i] = r * V[end,i]
        nres[i] = β * abs(V[end,i])
        if nres[i] < alg.tol
            converged += 1
        end
    end

    return values, vectors, T, ConvergenceInfo(converged, nres, residuals, 1, numops)
end

function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{ImplicitRestart})
    krylovdim = min(alg.krylovdim, length(x₀))
    dk = eltype(x₀) <: Real ? 2 : 1
    howmany + dk <= krylovdim || error("krylov dimension $(krylovdim) too small to compute $k eigenvalues")
    by, rev = eigsort(which)

    maxiter = alg.restart.maxiter

    # Compute arnoldi factorization
    iter = arnoldi(A, x₀, alg.orth; krylovdim = krylovdim)
    fact = start(iter)
    V = basis(fact)
    H = fact.H
    H0 = view(H, 1:krylovdim, 1:krylovdim) # square part
    numops = 0
    numiter = 0
    numconverged = 0
    residuals = similar(fact.V.basis, 0)
    sizehint!(residuals, howmany)
    while numiter < maxiter
        numiter += 1
        # Expand arnoldi factorization
        while fact.k != krylovdim
            next!(iter, fact)
            numops += 1
        end
        β = normres(fact)

        # Schur factorize into specified order
        T, U, values = schur(H0)
        p = sortperm(values, by = by, rev = rev)
        reorderschur!(T, p, U)

        # Also transform last row of the full H
        @inbounds for j=numconverged+1:krylovdim
            H[krylovdim+1,j] = β*U[krylovdim,j]
        end

        # Update V by applying U using Householder reflections
        for j = numconverged+1:krylovdim
            h, ν = householder(U, j:krylovdim, j)
            lmul!(U, h, j+1:krylovdim)
            rmulc!(V, h)
        end

        # Check convergence criterion and store residuals for converged schur vectors
        while numconverged < howmany
            abs(H[krylovdim+1,numconverged+1]) > alg.tol && break
            if eltype(H) <: Real && H[numconverged+2,numconverged+1] != 0
                abs(H[krylovdim+1,numconverged+2]) > alg.tol && break
                push!(residuals, H[krylovdim+1, numconverged+1]*V[krylovdim+1])
                push!(residuals, H[krylovdim+1, numconverged+2]*V[krylovdim+1])
                H[krylovdim+1, numconverged+1] = 0
                H[krylovdim+1, numconverged+2] = 0
                numconverged += 2
            else
                push!(residuals, H[krylovdim+1, numconverged+1]*V[krylovdim+1])
                H[krylovdim+1, numconverged+1] = 0
                numconverged += 1
            end
        end

        # Check stopping criterion
        numconverged >= howmany && break
        numiter == maxiter && break

        # Determine how many to keep
        keep = max(numconverged+dk, krylovdim >>1) # strictly smaller than krylovdim
        if eltype(H) <: Real && H[keep+1,keep] != 0 # we are in the middle of a 2x2 block
            keep -= 1 # conservative choice
        end

        # Shrink: update fact
        fact.k = keep
        # update fact.H
        @inbounds for j = numconverged+1:keep
            H[keep+1,j] = H[krylovdim+1,j]
        end
        for j = keep+1:krylovdim
            H[:,j] = 0
        end
        H[krylovdim+1,:] = 0

        V[keep+1] = V[krylovdim+1]
        while length(V) > keep+1
            pop!(V)
        end

        # Restore arnoldi form by bringing H into Hessenberg form (row operations)
        for j = keep:-1:numconverged+1
            h, ν = householder(H, j+1, numconverged+1:j, j)
            H[j+1,j] = ν
            @inbounds H[j+1,numconverged+1:j-1] = 0
            lmul!(H, h)
            rmulc!(H, h, 1:j)
            rmulc!(V, h)
        end
    end
    if numconverged > howmany
        howmany = numconverged
    else
        for j = numconverged+1:howmany
            push!(residuals,H[krylovdim+1,j]*V[krylovdim+1])
        end
    end

    Tf = getindex(H,1:howmany,1:howmany)
    T, _, values = hschur!(T)
    vectors = V[1:howmany]

    return values, vectors, T, ConvergenceInfo(numconverged, map(vecnorm, residuals), residuals, numiter, numops)
end
