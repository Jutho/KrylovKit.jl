function exponentiate(t::Number, A, v, alg::Lanczos{ExplicitRestart})
    # process initial vector and determine result type
    β = vecnorm(v)
    w = one(t)*apply(A,v/one(β)) # used to determine return type
    numops = 1
    scale!(w, v, 1/β)

    # initialize iterator
    if isa(alg, Lanczos)
        iter = LanczosIterator(A, w, alg.orth, true)
    else
        iter = ArnoldiIterator(A, w, alg.orth)
    end
    fact = start(iter)
    numops += 1

    # algorithm parameters
    sgn = sign(t)
    τ = abs(t)
    Δτ = τ

    η = oftype(normres(fact), alg.tol / τ) # tolerance per unit step
    if η < length(w)*eps(typeof(η))*10
        η = length(w)*eps(typeof(η))*10
        warning("tolerance too small, increasing to $(η*τ)")
    end
    totalerr = zero(η)
    krylovdim = alg.krylovdim
    maxiter = alg.restart.maxiter

    δ = 0.9 # safety factor
    γ = 1.2

    # start outer iteration loop
    numiter = 0
    while true
        numiter += 1
        Δτ = numiter == maxiter ? τ : min(Δτ, τ)

        # Lanczos or Arnoldi factorization
        while normres(fact) > η && fact.k < krylovdim
            fact = next!(iter, fact)
            numops += 1
        end
        K = fact.k # current Krylov dimension
        V = basis(fact)

        # Small matrix exponential and error estimation
        if isa(alg, Lanczos)
            T = matrix(fact)
            D, U = eig(T)

            # Estimate largest allowed time step
            ϵ = zero(η)
            while true
                ϵ₁ = zero(eltype(T))
                ϵ₂ = zero(eltype(T))
                @inbounds for k = 1:K
                    ϵ₁ += U[K,k] * exp(sgn * Δτ/2 * D[k]) * conj(U[1,k])
                    ϵ₂ += U[K,k] * exp(sgn * Δτ * D[k]) * conj(U[1,k])
                end
                ϵ = normres(fact) * ( 2*abs(ϵ₁)/3 + abs(ϵ₂)/6 ) # error per unit time: see Lubich

                if ϵ < δ * η || numiter == maxiter
                    break
                else # reduce time step
                    Δτ = signif(δ * (η / ϵ)^(1/krylovdim) * Δτ, 2)
                end
            end

            # Apply time step
            totalerr += Δτ * ϵ
            y = map(exp, (sgn*Δτ)*D)
            @inbounds for k = 1:length(y)
                y[k] *= conj(U[1,k])
            end
            y = U*y
        else
            # TODO: matrix exponential and error estimation in case of Arnoldi
            error("Arnoldi is not yet implemented")
        end

        # Finalize step
        A_mul_B!(w, V, y)
        τ -= Δτ

        if iszero(τ) # should always be true if numiter == maxiter
            scale!(w, β)
            converged = totalerr < alg.tol ? 1 : 0
            return w, ConvergenceInfo(converged, totalerr, nothing, numiter, numops)
        else
            normw = vecnorm(w)
            β *= normw
            scale!(w, inv(normw))
            fact = start!(iter, fact)
        end
    end
end
