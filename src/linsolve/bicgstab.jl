function linsolve(operator, b, x₀, alg::BiCGStab, a₀::Number=0, a₁::Number=1)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(inner(b, y₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = scale(b, one(T)) # r = mul!(similar(b, T), b, 1)
    r = iszero(α₀) ? r : add!!(r, x₀, -α₀)
    r = add!!(r, y₀, -α₁)
    x = scale!!(zerovector(r), x₀, 1)
    normr = norm(r)
    S = typeof(normr)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol::S = alg.tol
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr < tol
        if alg.verbosity > 0
            @info """BiCGStab linsolve converged without any iterations:
             *  norm of residual = $normr
             *  number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    end

    # First iteration
    numiter += 1
    r_shadow = scale!!(zerovector(r), r, 1)     # shadow residual
    ρ = inner(r_shadow, r)

    # Method fails if ρ is zero.
    if ρ ≈ 0.0
        @warn """BiCGStab linsolve errored after $numiter iterations:
        *   norm of residual = $normr
        *   number of operations = $numops"""
        return (x, ConvergenceInfo(0, r, normr, numiter, numops))
    end

    ## BiCG part of the algorithm.
    p = scale!!(zerovector(r), r, 1)
    v = apply(operator, p, α₀, α₁)
    numops += 1

    σ = inner(r_shadow, v)
    α = ρ / σ

    s = scale!!(zerovector(r), r, 1)
    s = add!!(s, v, -α) # half step residual

    xhalf = scale!!(zerovector(x), x, 1)
    xhalf = add!!(xhalf, p, +α) # half step iteration

    normr = norm(s)

    # Check for early return at half step.
    if normr < tol
        # Replace approximate residual with the actual residual.
        s = scale!!(zerovector(b), b, 1)
        s = add!!(s, apply(operator, xhalf, α₀, α₁), -1)
        numops += 1

        normr_act = norm(s)
        if normr_act < tol
            if alg.verbosity > 0
                @info """BiCGStab linsolve converged at iteration $(numiter-1/2):
                 *  norm of residual = $normr_act
                 *  number of operations = $numops"""
            end
            return (xhalf, ConvergenceInfo(1, s, normr_act, numiter, numops))
        end
    end

    ## GMRES part of the algorithm.
    t = apply(operator, s, α₀, α₁)
    numops += 1

    ω = inner(t, s) / inner(t, t)

    x = scale!!(x, xhalf, 1)
    x = add!!(x, s, +ω) # full step iteration

    r = scale!!(r, s, 1)
    r = add!!(r, t, -ω) # full step residual

    # Check for early return at full step.
    normr = norm(r)
    if normr < tol
        # Replace approximate residual with the actual residual.
        r = scale!!(r, b, 1)
        r = add!!(r, apply(operator, x, α₀, α₁), -1)
        numops += 1

        normr_act = norm(r)
        if normr_act < tol
            if alg.verbosity > 0
                @info """BiCGStab linsolve converged at iteration $(numiter):
                *  norm of residual = $normr_act
                *  number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr_act, numiter, numops))
        end
    end

    while numiter < maxiter
        if alg.verbosity > 0
            msg = "BiCGStab linsolve in iter $numiter: "
            msg *= "normres = "
            msg *= @sprintf("%12e", normr)
            @info msg
        end

        numiter += 1
        ρold = ρ
        ρ = inner(r_shadow, r)
        β = (ρ / ρold) * (α / ω)

        p = add!!(p, v, -ω)
        p = add!!(p, r, 1, β)

        v = apply(operator, p, α₀, α₁)
        numops += 1

        σ = inner(r_shadow, v)
        α = ρ / σ

        s = scale!!(s, r, 1)
        s = add!!(s, v, -α) # half step residual

        xhalf = scale!!(xhalf, x, 1)
        xhalf = add!!(xhalf, p, +α) # half step iteration

        normr = norm(s)

        if alg.verbosity > 0
            msg = "BiCGStab linsolve in iter $(numiter-1/2): "
            msg *= "normres = "
            msg *= @sprintf("%12e", normr)
            @info msg
        end

        # Check for return at half step.
        if normr < tol
            # Compute non-approximate residual.
            s = scale!!(zerovector(b), b, 1)
            s = add!!(s, apply(operator, xhalf, α₀, α₁), -1)
            numops += 1

            normr_act = norm(s)
            if normr_act < tol
                if alg.verbosity > 0
                    @info """BiCGStab linsolve converged at iteration $(numiter-1/2):
                    *  norm of residual = $normr_act
                    *  number of operations = $numops"""
                end
                return (xhalf, ConvergenceInfo(1, s, normr_act, numiter, numops))
            end
        end

        ## GMRES part of the algorithm.
        t = apply(operator, s, α₀, α₁)
        numops += 1

        ω = inner(t, s) / inner(t, t)

        x = scale!!(x, xhalf, 1)
        x = add!!(x, s, +ω) # full step iteration

        r = scale!!(r, s, 1)
        r = add!!(r, t, -ω) # full step residual

        # Check for return at full step.
        normr = norm(r)
        if normr < tol
            # Replace approximate residual with the actual residual.
            r = scale!!(r, b, 1)
            r = add!!(r, apply(operator, x, α₀, α₁), -1)
            numops += 1

            normr_act = norm(r)
            if normr_act < tol
                if alg.verbosity > 0
                    @info """BiCGStab linsolve converged at iteration $(numiter):
                    *  norm of residual = $normr_act
                    *  number of operations = $numops"""
                end
                return (x, ConvergenceInfo(1, r, normr_act, numiter, numops))
            end
        end
    end

    if alg.verbosity > 0
        @warn """BiCGStab linsolve finished without converging after $numiter iterations:
        *   norm of residual = $normr
        *   number of operations = $numops"""
    end
    return (x, ConvergenceInfo(0, r, normr, numiter, numops))
end
