function exponentiate(t::Number, A, v, alg::Lanczos)
    # process initial vector and determine result type
    β = vecnorm(v)
    Av = apply(A, v) # used to determine return type
    numops = 1
    T = promote_type(eltype(Av), typeof(β), typeof(t))
    S = real(T)
    w = similar(Av, T)
    scale!(w, v, 1/β)

    # krylovdim and related allocations
    krylovdim = min(alg.krylovdim, length(v))
    UU = Matrix{S}(krylovdim, krylovdim)
    yy1 = Vector{T}(krylovdim)
    yy2 = Vector{T}(krylovdim)

    # initialize iterator
    iter = LanczosIterator(A, w, alg.orth, true)
    fact = start(iter)
    numops += 1
    sizehint!(fact, krylovdim)

    # time step parameters
    sgn = sign(t)
    τ::S = abs(t)
    Δτ::S = τ

    # tolerance
    η::S = alg.tol / τ # tolerance per unit step
    if η < length(w)*eps(typeof(η))
        η = length(w)*eps(typeof(η))
        warn("tolerance too small, increasing to $(η*τ)")
    end
    totalerr = zero(η)

    δ::S = 0.9 # safety factor

    # start outer iteration loop
    maxiter = alg.maxiter
    numiter = 0
    while true
        numiter += 1
        Δτ = numiter == maxiter ? τ : min(Δτ, τ)

        # Lanczos or Arnoldi factorization
        while normres(fact) > η && length(fact) < krylovdim
            fact = next!(iter, fact)
            numops += 1
        end
        K = fact.k # current Krylov dimension
        V = basis(fact)
        m = length(fact)

        # Small matrix exponential and error estimation
        U = copy!(view(UU, 1:m, 1:m), I)
        H = rayleighquotient(fact) # tridiagonal
        D, U = eig!(H, U)

        # Estimate largest allowed time step
        ϵ::S = zero(η)
        while true
            ϵ₁ = zero(eltype(H))
            ϵ₂ = zero(eltype(H))
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
        y1 = view(yy1, 1:m)
        y2 = view(yy2, 1:m)
        @inbounds for k = 1:m
            y1[k] = exp(sgn*Δτ*D[k])*conj(U[1,k])
        end
        y2 = A_mul_B!(y2, U, y1)

        # Finalize step
        A_mul_B!(w, V, y2)
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
