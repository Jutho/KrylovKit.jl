function lssolve(operator, b, alg::LSMR, λ_::Real=0)
    # Initialisation: determine number type
    u₀ = b
    v₀ = apply_adjoint(operator, u₀)
    T = typeof(inner(v₀, v₀) / inner(u₀, u₀))
    u = scale(u₀, one(T))
    v = scale(v₀, one(T))
    β = norm(u)
    S = typeof(β)
    u = scale!!(u, 1 / β)
    v = scale!!(v, 1 / β)
    α = norm(v)
    v = scale!!(v, 1 / α)

    # Scalar variables for the bidiagonalization
    ᾱ = α
    ζ̄ = α * β
    ρ = one(S)
    θ = zero(S)
    ρ̄ = one(S)
    c̄ = one(S)
    s̄ = zero(S)

    absζ̄ = abs(ζ̄)

    # Vector variables
    x = zerovector(v)
    h = v
    h̄ = zerovector(v)

    r = scale(u, β)
    Ah = zerovector(u)
    Ah̄ = zerovector(u)

    # Algorithm parameters
    numiter = 0
    numops = 1 # One (adjoint) function application for v
    maxiter = alg.maxiter
    tol::S = alg.tol
    λ::S = convert(S, λ_)

    # Check for early return
    if abs(ζ̄) < tol
        if alg.verbosity > 0
            @info """LSMR lssolve converged without any iterations:
             *  ‖b - A * x ‖ = $β
             *  ‖[b - A * x; λ * x] ‖ = $β
             *  ‖ Aᴴ(b - A x) - λ^2 x ‖ = $absζ̄
             *  number of operations = $numops"""
        end
        return (x, ConvergenceInfo(1, r, abs(ζ̄), numiter, numops))
    end

    while true
        numiter += 1
        Av = apply_normal(operator, v)
        Ah = add!!(Ah, Av, 1, -θ / ρ)

        # βₖ₊₁ uₖ₊₁ = A vₖ - αₖ uₖ₊₁
        u = add!!(Av, u, -α, 1)
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
        θ̄ = s̄ * ρ # θ̄ₖ = s̄ₖ₋₁ * ρₖ
        c̄ρ = c̄ * ρ # c̄ₖ₋₁ * ρₖ
        ρ̄ = hypot(c̄ρ, θ) # ρ̄ₖ = sqrt((c̄ₖ₋₁ * ρₖ)^2 + θₖ₊₁^2)
        c̄ = c̄ρ / ρ̄ # c̄ₖ = c̄ₖ₋₁ * ρₖ / ρ̄ₖ
        s̄ = θ / ρ̄ # s̄ₖ = θₖ₊₁ / ρ̄ₖ
        ζ = c̄ * ζ̄ # ζₖ = c̄ₖ * ζ̄_{k}
        ζ̄ = -s̄ * ζ̄ # ζ̄ₖ₊₁ = -s̄ₖ * ζ̄ₖ

        # Update h, h̄, x
        h̄ = add!!(h̄, h, 1, -θ̄ * ρ / (ρold * ρ̄old)) # h̄ₖ = hₖ - θ̄ₖ * ρₖ / (ρₖ₋₁ * ρ̄ₖ₋₁) * h̄ₖ₋₁
        Ah̄ = add!!(Ah̄, Ah, 1, -θ̄ * ρ / (ρold * ρ̄old)) # h̄ₖ = hₖ - θ̄ₖ * ρₖ / (ρₖ₋₁ * ρ̄ₖ₋₁) * h̄ₖ₋₁

        x = add!!(x, h̄, ζ / (ρ * ρ̄)) # xₖ = xₖ₋₁ + ζₖ / (ρₖ * ρ̄ₖ) * h̄ₖ
        r = add!!(r, Ah̄, -ζ / (ρ * ρ̄)) # rₖ = rₖ₋₁ - ζₖ / (ρₖ * ρ̄ₖ) * Ah̄ₖ

        h = add!!(h, v, 1, -θ / ρ) # hₖ₊₁ = vₖ₊₁ - θₖ₊₁ / ρₖ * hₖ
        # Ah is updated in the next iteration when A v is computed

        absζ̄ = abs(ζ̄)
        if absζ̄ <= tol
            if alg.verbosity > 0
                @info """LSMR lssolve converged at iteration $numiter:
                 *  ‖ b - A x ‖ = $(norm(r))
                 *  ‖ x ‖ = $(norm(x))
                 *  ‖ Aᴴ(b - A x) - λ^2 x ‖ = $absζ̄
                 *  number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, absζ̄, numiter, numops))
        elseif numiter >= maxiter
            if alg.verbosity > 0
                normr = norm(r)
                normx = norm(x)
                @warn """LSMR lssolve finished without converging after $numiter iterations:
                 *  ‖ b - A x ‖ = $(norm(r))
                 *  ‖ x ‖ = $(norm(x))
                 *  ‖ Aᴴ(b - A x) - λ^2 x ‖ = $absζ̄
                 *  number of operations = $numops"""
            end
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
