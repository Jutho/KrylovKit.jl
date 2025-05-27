# 
function bieigsolve(f, v₀, w₀, howmany::Int, which::Selector, alg::BiArnoldi;
                    alg_rrule=alg)
    (S, Q), (T, Z), (βrV, βrW), fact, converged, numiter, numops = _schursolve(f, v₀, w₀, howmany,
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

    SS = view(S, 1:howmany′, 1:howmany′)
    TT = view(T, 1:howmany′, 1:howmany′)
    valuesS = schur2eigvals(SS)
    valuesT = schur2eigvals(TT)

    # Compute eigenvectors
    VS = view(Q, :, 1:howmany′) * schur2eigvecs(SS)
    VT = view(Z, :, 1:howmany′) * schur2eigvecs(TT)
    vectorsS = let B = basis(fact)[1]
        [B * v for v in cols(VS)]
    end
    vectorsT = let B = basis(fact)[2]
        [B * v for v in cols(VT)]
    end

    H, K = rayleighquotient(fact)
    residualsS = _getresiduals(βrV, H[end, :], VS)
    # residualsT = _getresiduals(βrW, rW, VT )

    # we need to match the eigenvalues; sometimes λ and λ* get mismatched,
    # if, e.g., one sorts by the real part
    matchperm = _geteigenspacematchperm!!(valuesS, valuesT, vectorsS, vectorsT, residualsS)


    normresidualsS = [abs(normres(fact)[1]) * abs(last(v)) for v in cols(VS)]
    normresidualsT = [abs(normres(fact)[2]) * abs(last(v)) for v in cols(VT)]

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

    resize!(valuesS, length(matchperm))
    resize!(vectorsS, length(matchperm))

    return valuesS, vectorsS, vectorsT[matchperm],
           ConvergenceInfo(converged, residualsS, max.(normresidualsS, normresidualsT),
                           numiter, numops)
end

function _getresiduals(βr, h, c)
    # || r_j || = ... = || v~_{l+1} || |h^*_l c_j |
    # where v~_{l+1} is the biorthogonality corrected residual, 
    #       h^*_l is the final term in the Arnoldi expansion and 
    #       c_j is the last Ritzvector
    # from the _schursolve call we get || v~_{l+1} || = βrV

    residuals = zeros(real(eltype(c)), size(c, 2))
    for j in axes(c, 2)
        residuals[j] = βr * abs(h'c[:, j])
    end
    residuals
end

function _geteigenspacematchperm!!(valuesS, valuesT, vectorsS, vectorsT, residualsS)
    matchperm = zeros(Int64, length(valuesS))
    usedvaluesT = zeros(Bool, length(valuesT))
    firstunusedT = 1
    for i in eachindex(matchperm)
        # as both arrays are sorted roughly similar, tracking the first valid index
        # changes the scaling of the loop from O(n^2) to rougly O(n)
        while usedvaluesT[firstunusedT]
            firstunusedT += 1
        end
        for j in firstunusedT:length(valuesT)
            usedvaluesT[j] && continue 

            if isapprox(norm(valuesS[i] - conj(valuesT[j])), 0.0; atol=max(norm(valuesS[i]), 1.0) * sqrt(eps(real(eltype(valuesS)))))
                overlapji = inner(vectorsT[j], vectorsS[i])
                if !isapprox(abs(overlapji), 0.0)
                    matchperm[i] = j
                    usedvaluesT[j] = true
                    # normalize and rotate the vectors according to biorthogonality,
                    #          <W_i | V_j> = delta_ij
                    # distribute the weight to both vectors
                    vectorsS[i] = scale!!(vectorsS[i], 1 / sqrt(overlapji))
                    vectorsT[j] = scale!!(vectorsT[j], 1 / conj(sqrt(overlapji)))
                    residualsS[i] /= abs(sqrt(overlapji))
                    break
                end
            end
        end

        if matchperm[i] == 0
            resize!(matchperm, i - 1)
            @error "BiArnoldi bieigsolve converged with mismatched left- and right-eigenspaces"
            break
        end
    end

    return matchperm
end

function _schursolve(f, v₀, w₀, howmany::Int, which::Selector, alg::BiArnoldi)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # initialize arnoldi factorization
    iter = BiArnoldiIterator(f, v₀, w₀, alg.orth)
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
    Wvv = zeros(eltype(fact), krylovdim)
    Vww = zeros(eltype(fact), krylovdim)

    MM = fill(zero(eltype(fact)), krylovdim, krylovdim)
    temp = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize storage
    L = length(fact) # == 1
    V, W = basis(fact)
    MM[L, L] = inner(W[L], V[L])
    converged = 0
    βrV = βrW = 0.0
    local S, T, Q, Z
    while true
        βv, βw = normres(fact)
        L = length(fact)

        if (βv <= tol || βw <= tol) && L < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $L (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end
        end
        if L == krylovdim || (βv <= tol && βw <= tol) || (alg.eager && L >= howmany) # process

            # Step 1
            H = view(HH, 1:L, 1:L)
            K = view(KK, 1:L, 1:L)
            Q = view(QQ, 1:L, 1:L)
            Z = view(ZZ, 1:L, 1:L)
            M = view(MM, 1:L, 1:L)
            h = view(HH, L + 1, 1:L)
            k = view(KK, L + 1, 1:L)
            Wv = view(Wvv, 1:L)
            Vw = view(Vww, 1:L)

            copyto!(Q, I)
            copyto!(Z, I)
            copyto!.((H, K), rayleighquotient(fact))

            rV, rW = residual(fact)
            rV = scale!!(rV, 1 / βv)
            rW = scale!!(rW, 1 / βw)

            # Step 2 and 3 - Correct H, K and the residuals using the oblique projection

            # Compute the projections W* residual(V) and V* residual(W)
            V, W = basis(fact)
            for i in eachindex(Wv)
                Wv[i] = inner(W[i], rV)
                Vw[i] = inner(V[i], rW)
            end

            F = lu(M)
            MWv = F \ Wv
            MVw = F' \ Vw
            add!(view(H, :, L), MWv, βv)
            add!(view(K, :, L), MVw, βw)

            for i in eachindex(Wv)
                rV = add!!(rV, V[i], -MWv[i])
                rW = add!!(rW, W[i], -MVw[i])
            end

            # Step 5 - Compute dense schur factorization
            S, Q, valuesH = hschur!(H, Q)
            T, Z, valuesK = hschur!(K, Z)

            # Step 6 - Order the Schur decompositions
            by, rev = eigsort(which)
            pH = sortperm(valuesH; by=by, rev=rev)
            pK = sortperm(valuesK; by=by ∘ conj, rev=rev)

            S, Q = permuteschur!(S, Q, pH)
            T, Z = permuteschur!(T, Z, pK)

            # Partially Step 7 & 8 - Correction of hm and km
            h = mul!(h, view(Q, L, :), βv)
            k = mul!(k, view(Z, L, :), βw)

            βrV = norm(rV)
            βrW = norm(rW)

            converged = 0
            while converged < length(fact)
                # The authors suggest the convergence should also include the 
                # 1. a biorthogonality component, i.e., kappa_j / |rho_j| in the paper 
                #    with kappa_j = norm(w_j* v_j) and rho_j = abs(w_j* A v_j) / kappa_j 
                # 2. a contribution of the norms of tilde v and tilde w

                # For the first case (1.), we use the Ritz values instead of the Rayleigh quotients 
                # as suggested by the authors 

                # This is Eq. 10 in the paper
                xh = abs(h[converged + 1]) 
                xk = abs(k[converged + 1]) 
                if max(xh, xk) <= tol
                    converged += 1
                else
                    break
                end
            end
            if eltype(T) <: Real &&
               0 < converged < length(fact) &&
               T[converged + 1, converged] != 0
                converged -= 1
            end

            if converged >= howmany || (βv <= tol && βw <= tol)
                break
            elseif alg.verbosity >= EACHITERATION_LEVEL
                @info "Arnoldi schursolve in iteration $numiter, step = $L: $converged values converged, normres = $(normres2string(abs.(h[1:howmany])))"
            end
        end

        if L < krylovdim # expand
            fact = expand!(iter, fact; verbosity=alg.verbosity)
            V, W = basis(fact)

            # update M with the new basis vectors
            for i in 1:L
                MM[i, L + 1] = inner(W[i], V[L + 1])
                MM[L + 1, i] = inner(W[L + 1], V[i])
            end
            MM[L + 1, L + 1] = inner(W[L + 1], V[L + 1])

            numops += 1
        else # shrink
            numiter == maxiter && break

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            while eltype(H) <: Real && (H[keep + 1, keep] != 0 || K[keep + 1, keep] != 0)
                # we are in the middle of a 2x2 block; this cannot happen if keep == converged, so we can decrease keep
                # however, we have to make sure that we do not end up with keep = 0
                if keep > 1
                    keep -= 1 # conservative choice
                else
                    keep += 1
                    if krylovdim == 2
                        alg.verbosity >= WARN_LEVEL &&
                            @warn "Arnoldi iteration got stuck in a 2x2 block, consider increasing the Krylov dimension"
                        break
                    end
                end
            end

            # Setp 10 & 11 - Correct the kept part of H and K and the residual 

            # We know that 
            #   Vm* residual(V) = Q_1* (Vl* residual(V)) = Q_1* Vl* (residual - Vl Ml^-1 Wl* residual) = -Q_1* Ml^-1 Wl* residual = -Q_1* MWv 
            # as Vl*Vl = Id and Vl* residual = 0

            Vv = -adjoint(Q[:, 1:keep]) * MWv
            Ww = -adjoint(Z[:, 1:keep]) * MVw

            H[1:keep, 1:keep] += Vv * transpose(h[1:keep])
            K[1:keep, 1:keep] += Ww * transpose(k[1:keep])

            # newresidual = (I - Vm Vm*) oldresidual = (I - Vl Q1 Vm*) oldresidual = oldresidual + Vl Q1 Q_1^* MWv = oldresidual + Vl Q1 Vv
            Q1Vv = Q[:, 1:keep] * Vv
            Z1Ww = Z[:, 1:keep] * Ww

            V, W = basis(fact)
            for i in eachindex(Q1Vv)
                rV = add!!(rV, V[i], -Q1Vv[i])
                rW = add!!(rW, W[i], -Z1Ww[i])
            end

            βpv = norm(rV)
            βpw = norm(rW)

            h .*= βpv
            k .*= βpw

            # Restore Arnoldi form in the first keep columns; this is not part of the original paper
            _restorearnoldiformandupdatebasis!(keep, H, Q, h, rayleighquotient(fact)[1],
                                               V, rV, βpv)
            _restorearnoldiformandupdatebasis!(keep, K, Z, k, rayleighquotient(fact)[2],
                                               W, rW, βpw)

            # Update M according to the transformation M -> Z'MQ to save some inner products later
            _M = view(MM, 1:keep, 1:keep)
            _temp = view(temp, 1:keep, 1:L)
            mul!(_temp, (Z[:, 1:keep])', M)
            mul!(_M, _temp, Q[:, 1:keep])

            # Shrink Arnoldi factorization
            fact = shrink!(fact, keep; verbosity=alg.verbosity)
            numiter += 1
        end
    end

    return (S, Q), (T, Z), (βrV, βrW), fact, converged, numiter, numops
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