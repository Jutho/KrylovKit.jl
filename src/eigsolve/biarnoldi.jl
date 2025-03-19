function eigsolve(A, AH, v₀, w₀, howmany::Int, which::Selector, alg::BiArnoldi;
                  alg_rrule=alg)
    S, Q, T, Z, fact, converged, numiter, numops = _schursolve(A, AH, v₀, w₀, howmany,
                                                               which, alg)

    howmany′ = howmany
    if eltype(T) <: Real && howmany < length(fact) && T[howmany + 1, howmany] != 0
        howmany′ += 1
    elseif size(T, 1) < howmany
        howmany′ = size(T, 1)
    end
    if converged > howmany
        howmany′ = converged
    end

    TT = view(T, 1:howmany′, 1:howmany′)
    SS = view(S, 1:howmany′, 1:howmany′)
    valuesT = schur2eigvals(TT)
    valuesS = schur2eigvals(SS)

    # Compute eigenvectors
    VS = view(Q, :, 1:howmany′) * schur2eigvecs(SS)
    VT = view(Z, :, 1:howmany′) * schur2eigvecs(TT)
    vectorsS = let B = basis(fact)[1]
        [B * v for v in cols(VS)]
    end
    vectorsT = let B = basis(fact)[2]
        [B * v for v in cols(VT)]
    end

    residualsS = let r = residual(fact)[1]
        [scale(r, last(v)) for v in cols(VS)]
    end
    residualsT = let r = residual(fact)[2]
        [scale(r, last(v)) for v in cols(VT)]
    end

    normresidualsS = [normres(fact)[1] * abs(last(v)) for v in cols(VS)]
    normresidualsT = [normres(fact)[2] * abs(last(v)) for v in cols(VT)]

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """Arnoldi eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresidualsS))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """Arnoldi eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresidualsS))
        * number of operations = $numops"""
    end
    return (valuesS, valuesT),
           (vectorsS, vectorsT),
           (ConvergenceInfo(converged, residualsS, normresidualsS, numiter, numops), ConvergenceInfo(converged, residualsT, normresidualsT, numiter, numops))
end

function _schursolve(A, AH, v₀, w₀, howmany::Int, which::Selector, alg::BiArnoldi)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize arnoldi factorization
    iter = BiArnoldiIterator(A, AH, v₀, w₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity)
    numops = 1
    sizehint!(fact, krylovdim)
    βv, βw = normres(fact)
    tol::eltype(βv) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    KK = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    QQ = fill(zero(eltype(fact)), krylovdim, krylovdim)
    ZZ = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize storage
    K = length(fact) # == 1
    converged = 0
    local S, T, Q, Z
    while true
        βv, βw = normres(fact)
        K = length(fact)

        if (βv <= tol || βw <= tol) && K < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end
        end
        if K == krylovdim || (βv <= tol && βw <= tol) || (alg.eager && K >= howmany) # process

            # Step 1
            _H = view(HH, 1:K, 1:K)
            _K = view(KK, 1:K, 1:K)
            Q = view(QQ, 1:K, 1:K)
            Z = view(ZZ, 1:K, 1:K)
            _h = view(HH, K + 1, 1:K)
            _k = view(KK, K + 1, 1:K)

            copyto!(Q, I)
            copyto!(Z, I)

            copyto!(_H, rayleighquotient(fact)[1])
            copyto!(_K, rayleighquotient(fact)[2])

            rV, rW = residual(fact)
            normalize!(rV)
            normalize!(rW)

            # Compute the oblique Projection
            # TODO: This should reuse the old projection after restarting
            M = fill(zero(eltype(fact)), krylovdim, krylovdim)
            for i in axes(M, 1)
                for j in axes(M, 2)
                    M[i, j] = dot(fact.W[i], fact.V[j])
                end
            end

            # Step 2 and 3 - Correct H, K and the residuals using the oblique projection

            # Compute the projections W* residual(V) and V* residual(W)
            Wv = zeros(krylovdim)
            Vw = zeros(krylovdim)
            for i in eachindex(Wv)
                Wv[i] = dot(fact.W[i], rV)
                Vw[i] = dot(fact.V[i], rW)
            end

            MWv = inv(M) * Wv
            MVw = inv(M') * Vw

            _H[:, end] += MWv * βv
            _K[:, end] += MVw * βw

            for i in eachindex(Wv)
                rV -= fact.V[i] * MWv[i]
                rW -= fact.W[i] * MVw[i]
            end

            # Step 5 - Compute dense schur factorization
            S, Q, valuesH = hschur!(_H, Q)
            T, Z, valuesK = hschur!(_K, Z)

            # Step 6 - Order the Schur decompositions
            by, rev = eigsort(which)
            pH = sortperm(valuesH; by=by, rev=rev)
            pK = sortperm(valuesK; by=by, rev=rev)

            S, Q = permuteschur!(S, Q, pH)
            T, Z = permuteschur!(T, Z, pK)

            # Partially Step 7 & 8 - Correction of hm and km
            _h = mul!(_h, view(Q, K, :), βv)
            _k = mul!(_k, view(Z, K, :), βw)

            converged = 0
            # TODO: this needs refinement
            while converged < length(fact) && abs(_h[converged + 1]) <= tol &&
                      abs(_k[converged + 1]) <= tol
                converged += 1
            end
            if eltype(T) <: Real &&
               0 < converged < length(fact) &&
               T[converged + 1, converged] != 0
                converged -= 1
            end

            if converged >= howmany || (βv <= tol && βw <= tol)
                break
            elseif alg.verbosity >= EACHITERATION_LEVEL
                @info "Arnoldi schursolve in iteration $numiter, step = $K: $converged values converged, normres = $(normres2string(abs.(_h[1:howmany])))"
            end
        end

        if K < krylovdim # expand
            fact = expand!(iter, fact; verbosity=alg.verbosity)
            numops += 1
        else # shrink
            numiter == maxiter && break

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
            while keep < krylovdim &&
                (isapprox(valuesH[pH[keep]], conj(valuesH[pH[keep + 1]]); rtol=1e-3) ||
                 isapprox(valuesK[pK[keep]], conj(valuesK[pK[keep + 1]]); rtol=1e-3))
                # increase the number of eigenvalues kept such that the eigenspace is not fractured at a eigenvalue pair
                keep += 1
            end

            # Setp 10 & 11 - Correct the kept part of H and K and the residual 

            # We know that 
            #   Vm* residual(V) = Q_1* (Vl* residual(V)) = Q_1* Vl* (residual - Vl Ml^-1 Wl* residual) = -Q_1* Ml^-1 Wl* residual = -Q_1* MWv 
            # as Vl*Vl = Id and Vl* residual = 0

            Vv = -adjoint(Q[:, 1:keep]) * MWv
            Ww = -adjoint(Z[:, 1:keep]) * MVw

            _H[1:keep, 1:keep] += Vv * _h[1:keep]'
            _K[1:keep, 1:keep] += Ww * _k[1:keep]'

            # newresidual = (I - Vm Vm*) oldresidual = (I - Vl Q1 Vm*) oldresidual = oldresidual + Vl Q1 Q_1^* MWv = oldresidual + Vl Q1 Vv
            Q1Vv = Q[:, 1:keep] * Vv
            Z1Ww = Z[:, 1:keep] * Ww

            for i in eachindex(Q1Vv)
                rV -= fact.V[i] * Q1Vv[i]
                rW -= fact.W[i] * Z1Ww[i]
            end

            βpv = norm(rV)
            βpw = norm(rW)

            _h .*= βpv
            _k .*= βpw

            # Restore Arnoldi form in the first keep columns; this is not part of the original paper
            _restorearnoldiformandupdatebasis!(keep, _H, Q, _h, rayleighquotient(fact)[1],
                                               fact.V, rV, βpv)
            _restorearnoldiformandupdatebasis!(keep, _K, Z, _k, rayleighquotient(fact)[2],
                                               fact.W, rW, βpw)

            # Shrink Arnoldi factorization
            fact = shrink!(fact, keep; verbosity=alg.verbosity)
            numiter += 1
        end
    end

    return S, Q, T, Z, fact, converged, numiter, numops
end

function _restorearnoldiformandupdatebasis!(keep, H, U, f, rq, B, r, βr)
    @inbounds for j in 1:keep
        H[keep + 1, j] = f[j]
    end
    @inbounds for j in keep:-1:1
        h, ν = householder(H, j + 1, 1:j, j)
        H[j + 1, j] = ν
        H[j + 1, 1:(j - 1)] .= 0
        lmul!(h, H)
        rmul!(view(H, 1:j, :), h')
        rmul!(U, h')
    end
    copyto!(rq, H) # copy back into fact

    # Update B by applying U
    basistransform!(B, view(U, :, 1:keep))
    return B[keep + 1] = scale!!(r, 1 / βr)
end