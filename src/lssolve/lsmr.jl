function lssolve(operator, b, alg::LSMR, λ_::Real=0)
    # Initial function operation and division defines number type
    x₀ = apply_adjoint(operator, b)
    T = typeof(inner(x₀, x₀) / inner(b, b))
    r = scale(b, one(T))
    β = norm(r)
    x = scale(x₀, zero(T))
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol::S = alg.tol
    λ::S = convert(S, λ_)

    # Initialisation
    numiter = 0
    numops = 1 # operator has been applied once to determine x₀
    u = scale!!(r, 1 / β)
    v = apply_adjoint(operator, u)
    numops += 1
    α = norm(v)
    v = scale!!(v, 1 / α)
    ᾱ = α
    ζ̄ = α * β
    ρ = one(S)
    ρ̄ = one(S)
    c̄ = one(S)
    s̄ = zero(S)

    h = v
    h̄ = zerovector(x)

    # Initialize variables for estimation of ‖r‖.
    β̈ = β
    β̇ = zero(S)
    ρ̇ = one(S)
    τ̃ = zero(S)
    θ̃ = zero(S)
    ζ = zero(S)
    d = zero(S)

    normr = β
    normr̄ = β
    absζ̄ = abs(ζ̄)

    # Check for early return
    if abs(ζ̄) < tol
        if alg.verbosity > 0
            @info """LSMR lssolve converged without any iterations:
             *  ‖b - A * x ‖ = $β
             *  ‖[b - A * x; λ * x] ‖ = $β
             *  ‖ Aᴴ(b - A x) - λ^2 x ‖ = $absζ̄
             *  number of operations = $numops"""
        end
        return (x, ConvergenceInfo(1, scale(u, normr), abs(ζ̄), numiter, numops))
    end

    while true
        numiter += 1
        # βₖ₊₁ uₖ₊₁ = A vₖ - αₖ uₖ₊₁
        u = add!!(apply_normal(operator, v), u, -α, 1)
        β = norm(u)
        u = scale!!(u, 1 / β)
        # αₖ₊₁ vₖ₊₁ = Aᴴ uₖ₊₁ - βₖ₊₁ vₖ
        v = add!!(apply_adjoint(operator, u), v, -β, 1)
        α = norm(v)
        v = scale!!(v, 1 / α)
        numops += 2

        # Construct rotation P̂ₖ
        α̂ = hypot(ᾱ, λ) # α̂ₖ = sqrt(ᾱₖ^2 + λ^2)
        ĉ = ᾱ / α̂ # ĉ = ᾱₖ / α̂ₖ
        ŝ = λ / α̂ # ŝₖ = λ / α̂ₖ

        # Use a plane rotation Pₖ to turn Bₖ to Rₖ
        ρold = ρ # ρₖ₋₁
        ρ = hypot(α̂, β) # ρₖ
        c = α̂ / ρ # cₖ = α̂ₖ / ρₖ
        s = β / ρ # sₖ = βₖ₊₁ / ρₖ
        θ = s * α # θₖ₊₁ = sₖ * αₖ₊₁
        ᾱ = c * α # ᾱₖ₊₁ = cₖ * αₖ₊₁

        # Use a plane rotation P̄ₖ to turn Rₖᵀ to R̄ₖ
        ρ̄old = ρ̄ # ρ̄ₖ₋₁
        ζold = ζ # ζₖ₋₁
        θ̄ = s̄ * ρ # θ̄ₖ = s̄ₖ₋₁ * ρₖ
        c̄ρ = c̄ * ρ # c̄ₖ₋₁ * ρₖ
        ρ̄ = hypot(c̄ρ, θ) # ρ̄ₖ = sqrt((c̄ₖ₋₁ * ρₖ)^2 + θₖ₊₁^2)
        c̄ = c̄ρ / ρ̄ # c̄ₖ = c̄ₖ₋₁ * ρₖ / ρ̄ₖ
        s̄ = θ / ρ̄ # s̄ₖ = θₖ₊₁ / ρ̄ₖ
        ζ = c̄ * ζ̄ # ζₖ = c̄ₖ * ζ̄_{k}
        ζ̄ = -s̄ * ζ̄ # ζ̄ₖ₊₁ = -s̄ₖ * ζ̄ₖ

        # Update h, h̄, x
        h̄ = add!!(h̄, h, 1, -θ̄ * ρ / (ρold * ρ̄old)) # h̄ₖ = hₖ - θ̄ₖ * ρₖ / (ρₖ₋₁ * ρ̄ₖ₋₁) * h̄ₖ₋₁
        x = add!!(x, h̄, ζ / (ρ * ρ̄)) # xₖ = xₖ₋₁ + ζₖ / (ρₖ * ρ̄ₖ) * h̄ₖ
        h = add!!(h, v, 1, -θ / ρ) # hₖ₊₁ = vₖ₊₁ - θₖ₊₁ / ρₖ * hₖ

        # Estimate of ‖r‖
        #-----------------
        # Apply rotation P̂ₖ
        β́ = ĉ * β̈ # β́ₖ = ĉₖ * β̈ₖ
        β̌ = -ŝ * β̈ # β̌ₖ = -ŝₖ * β̈ₖ

        # Apply rotation Pₖ
        β̂ = c * β́ # β̂ₖ = cₖ * β́ₖ
        β̈ = -s * β́ # β̈ₖ₊₁ = -sₖ * β́ₖ

        # Construct and apply rotation P̃ₖ₋₁
        ρ̃ = hypot(ρ̇, θ̄) # ρ̃ₖ₋₁ = sqrt(ρ̇ₖ₋₁^2 + θ̄ₖ^2)
        c̃ = ρ̇ / ρ̃ # c̃ₖ₋₁ = ρ̇ₖ₋₁ / ρ̃ₖ₋₁
        s̃ = θ̄ / ρ̃ # s̃ₖ = θ̄ₖ / ρ̃ₖ₋₁
        θ̃old = θ̃ # θ̃ₖ₋₁
        θ̃ = s̃ * ρ̄ # θ̃ₖ = s̃ₖ₋₁ * ρ̄ₖ
        ρ̇ = c̃ * ρ̄ # ρ̇ₖ = c̃ₖ₋₁ * ρ̄ₖ
        β̇ = -s̃ * β̇ + c̃ * β̂ # β̇ₖ = -s̃ₖ * β̇ₖ₋₁ + c̃ₖ₋₁ * β̂ₖ

        # Update t̃ by forward substitution
        τ̃ = (ζold - θ̃old * τ̃) / ρ̃ # τ̃ₖ₋₁ = (ζₖ₋₁ - θ̃ₖ₋₁ * τ̃ₖ₋₂) / ρ̃ₖ₋₁
        τ̇ = (ζ - θ̃ * τ̃) / ρ̇ # τ̇ₖ = (ζₖ - θ̃ₖ * τ̃ₖ₋₁) / ρ̇ₖ

        # Compute ‖r‖ and ‖r̄‖
        sqrtd = hypot(d, β̌)
        normr = hypot(β̇ - τ̇, β̈)
        normr̄ = hypot(sqrtd, normr)

        absζ̄ = abs(ζ̄)
        if absζ̄ <= tol
            if alg.verbosity > 0
                @info """LSMR lssolve converged at iteration $numiter:
                 *  ‖ b - A x ‖ = $normr
                 *  ‖ [b - A x; λ x] ‖ = $normr̄
                 *  ‖ Aᴴ(b - A x) - λ^2 x ‖ = $absζ̄
                 *  number of operations = $numops"""
            end
            # TODO: r can probably be determined and updated along the way
            r = add!!(apply_normal(operator, x), b, 1, -1)
            numops += 1
            return (x, ConvergenceInfo(1, r, absζ̄, numiter, numops))
        elseif numiter >= maxiter
            if alg.verbosity > 0
                @warn """LSMR lssolve finished without converging after $numiter iterations:
                 *  ‖ b - A x ‖ = $normr
                 *  ‖ [b - A x; λ x] ‖ = $normr̄
                 *  ‖ Aᴴ(b - A x) - λ^2 x ‖ = $absζ̄
                 *  number of operations = $numops"""
            end
            r = add!!(apply_normal(operator, x), b, 1, -1)
            numops += 1
            return (x, ConvergenceInfo(0, r, absζ̄, numiter, numops))
        end
        if alg.verbosity > 1
            msg = "LSMR lssolve in iter $numiter: "
            msg *= "convergence measure ‖ Aᴴ(b - A x) - λ^2 x ‖ = "
            msg *= @sprintf("%.12e", absζ̄)
            @info msg
        end
    end
end
