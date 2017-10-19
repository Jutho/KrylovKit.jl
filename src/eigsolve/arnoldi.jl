# Arnoldi methods for eigenvalue problems
function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{NoRestart})
    krylovdim = min(alg.krylovdim, length(x₀))
    if eltype(x₀) <: Real
        howmany < krylovdim-1 || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    else
        howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end

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
    permuteschur!(T, U, p)
    normresschurvec = β*map(v->abs(last(v)), cols(U))
    converged = 0
    while converged < length(fact) && normresschurvec[converged+1] < tol
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
    if eltype(x₀) <: Real
        howmany < krylovdim-1 || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    else
        howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute arnoldi factorization
    iter1 = ArnoldiIterator(A, x₀, alg.orth)
    fact = start(iter1)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    numops = 1
    while length(fact) < krylovdim
        fact = next!(iter1, fact)
        numops += 1
        normres(fact) < tol && break
    end

    # Process
    # allocate storage
    HH = zeros(eltype(fact), krylovdim, krylovdim)
    UU = zeros(eltype(fact), krylovdim, krylovdim)

    # initialize
    β = normres(fact)
    m = length(fact)
    H = view(HH, 1:m, 1:m)
    U = view(UU, 1:m, 1:m)
    copy!(U, I)
    copy!(H, matrix(fact))

    # compute dense schur factorization
    T, U, values = hschur!(H, U)
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    permuteschur!(T, U, p)
    normresschurvec = β*map(v->abs(last(v)), cols(U))
    converged = 0
    while converged < length(fact) && normresschurvec[converged+1] < tol
        converged += 1
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter
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
        copy!(U, I)
        copy!(H, matrix(fact))

        # compute dense schur factorization
        T, U, values = hschur!(H, U)
        by, rev = eigsort(which)
        p = sortperm(values, by = by, rev = rev)
        permuteschur!(T, U, p)
        normresschurvec = β*map(v->abs(last(v)), cols(U))
        converged = 0
        while converged < length(fact) && normresschurvec[converged+1] < tol
            converged += 1
        end

        if converged >= howmany
            break
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
    normreseigvecs = β*map(v->abs(last(v)), cols(V))

    return values, vectors, ConvergenceInfo(converged, normreseigvecs, residuals, numiter, numops)
end


# function eigsolve(A, x₀, howmany::Int, which::Symbol, alg::Arnoldi{ImplicitRestart})
#     krylovdim = min(alg.krylovdim, length(x₀))
#     maxiter = alg.restart.maxiter
#     if eltype(x₀) <: Real
#         howmany < krylovdim-1 || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
#     else
#         howmany < krylovdim || error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
#     end
#
#     # Compute arnoldi factorization
#     iter = ArnoldiIterator(A, x₀, alg.orth)
#     fact = start(iter)
#     tol = oftype(normres(fact), alg.tol)
#     numops = 1
#     numiter = 1
#     while length(fact) < krylovdim
#         fact = next!(iter, fact)
#         numops += 1
#         normres(fact) < tol && break
#     end
#
#     H = zeros(matrix(fact), )
#     H = copy(matrix(fact)) # creates a Matrix copy
#     T, U, values = hschur!(H, one(H))
#     by, rev = eigsort(which)
#     p = sortperm(values, by = by, rev = rev)
#     permuteschur!(T, U, p)
#
#
#
#
#
#
#     krylovdim = min(alg.krylovdim, length(x₀))
#     dk = eltype(x₀) <: Real ? 2 : 1
#     howmany + dk <= krylovdim || error("krylov dimension $(krylovdim) too small to compute $k eigenvalues")
#     by, rev = eigsort(which)
#
#     maxiter = alg.restart.maxiter
#
#     # Compute arnoldi factorization
#     iter = arnoldi(A, x₀, alg.orth; krylovdim = krylovdim)
#     fact = start(iter)
#     V = basis(fact)
#     H = fact.H
#     H0 = view(H, 1:krylovdim, 1:krylovdim) # square part
#     numops = 0
#     numiter = 0
#     numconverged = 0
#     residuals = similar(fact.V.basis, 0)
#     sizehint!(residuals, howmany)
#     while numiter < maxiter
#         numiter += 1
#         # Expand arnoldi factorization
#         while fact.k != krylovdim
#             next!(iter, fact)
#             numops += 1
#         end
#         β = normres(fact)
#
#         # Schur factorize into specified order
#         T, U, values = schur(H0)
#         p = sortperm(values, by = by, rev = rev)
#         reorderschur!(T, p, U)
#
#         # Also transform last row of the full H
#         @inbounds for j=numconverged+1:krylovdim
#             H[krylovdim+1,j] = β*U[krylovdim,j]
#         end
#
#         # Update V by applying U using Householder reflections
#         for j = numconverged+1:krylovdim
#             h, ν = householder(U, j:krylovdim, j)
#             lmul!(U, h, j+1:krylovdim)
#             rmulc!(V, h)
#         end
#
#         # Check convergence criterion and store residuals for converged schur vectors
#         while numconverged < howmany
#             abs(H[krylovdim+1,numconverged+1]) > alg.tol && break
#             if eltype(H) <: Real && H[numconverged+2,numconverged+1] != 0
#                 abs(H[krylovdim+1,numconverged+2]) > alg.tol && break
#                 push!(residuals, H[krylovdim+1, numconverged+1]*V[krylovdim+1])
#                 push!(residuals, H[krylovdim+1, numconverged+2]*V[krylovdim+1])
#                 H[krylovdim+1, numconverged+1] = 0
#                 H[krylovdim+1, numconverged+2] = 0
#                 numconverged += 2
#             else
#                 push!(residuals, H[krylovdim+1, numconverged+1]*V[krylovdim+1])
#                 H[krylovdim+1, numconverged+1] = 0
#                 numconverged += 1
#             end
#         end
#
#         # Check stopping criterion
#         numconverged >= howmany && break
#         numiter == maxiter && break
#
#         # Determine how many to keep
#         keep = max(numconverged+dk, krylovdim >>1) # strictly smaller than krylovdim
#         if eltype(H) <: Real && H[keep+1,keep] != 0 # we are in the middle of a 2x2 block
#             keep -= 1 # conservative choice
#         end
#
#         # Shrink: update fact
#         fact.k = keep
#         # update fact.H
#         @inbounds for j = numconverged+1:keep
#             H[keep+1,j] = H[krylovdim+1,j]
#         end
#         for j = keep+1:krylovdim
#             H[:,j] = 0
#         end
#         H[krylovdim+1,:] = 0
#
#         V[keep+1] = V[krylovdim+1]
#         while length(V) > keep+1
#             pop!(V)
#         end
#
#         # Restore arnoldi form by bringing H into Hessenberg form (row operations)
#         for j = keep:-1:numconverged+1
#             h, ν = householder(H, j+1, numconverged+1:j, j)
#             H[j+1,j] = ν
#             @inbounds H[j+1,numconverged+1:j-1] = 0
#             lmul!(H, h)
#             rmulc!(H, h, 1:j)
#             rmulc!(V, h)
#         end
#     end
#     if numconverged > howmany
#         howmany = numconverged
#     else
#         for j = numconverged+1:howmany
#             push!(residuals,H[krylovdim+1,j]*V[krylovdim+1])
#         end
#     end
#
#     Tf = getindex(H,1:howmany,1:howmany)
#     T, _, values = hschur!(T)
#     vectors = V[1:howmany]
#
#     return values, vectors, T, ConvergenceInfo(numconverged, map(vecnorm, residuals), residuals, numiter, numops)
# end
