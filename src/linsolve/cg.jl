function linsolve(operator, b, x₀, alg::CG, a₀::Real = 0, a₁::Real = 1; alg_rrule = alg)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(inner(b, y₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = scale(b, one(T))
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
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """CG linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "CG linsolve starts with norm of residual = $(normres2string(normr))"
    end

    # First iteration
    ρ = normr^2
    p = scale!!(zerovector(r), r, 1)
    q = apply(operator, p, α₀, α₁)
    α = ρ / inner(p, q)
    x = add!!(x, p, +α)
    r = add!!(r, q, -α)
    normr = norm(r)
    ρold = ρ
    ρ = normr^2
    β = ρ / ρold
    numops += 1
    numiter += 1

    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """CG linsolve converged at iteration $numiter:
            * norm of residual = $(normres2string(normr))
            * number of operations = $numops"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    end
    if alg.verbosity >= EACHITERATION_LEVEL
        @info "CG linsolve in iteration $numiter: normres = $(normres2string(normr))"
    end

    # Check for early return
    normr < tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while true
        p = add!!(p, r, 1, β)
        q = apply(operator, p, α₀, α₁)
        α = ρ / inner(p, q)
        x = add!!(x, p, α)
        r = add!!(r, q, -α)
        normr = norm(r)
        if normr < tol # recompute to account for buildup of floating point errors
            r = scale!!(r, b, 1)
            r = add!!(r, apply(operator, x, α₀, α₁), -1)
            normr = norm(r)
            ρ = normr^2
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = normr^2
            β = ρ / ρold
        end
        numops += 1
        numiter += 1
        if normr < tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info """CG linsolve converged at iteration $numiter:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """CG linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, normr, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "CG linsolve in iteration $numiter: normres = $(normres2string(normr))"
        end
    end
    return
end
