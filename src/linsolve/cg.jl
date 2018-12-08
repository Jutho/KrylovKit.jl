function linsolve(operator, b, x₀, alg::CG, a₀::Real = 0, a₁::Real = 1)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(dot(b, y₀)/norm(b)*one(a₀)*one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = copyto!(similar(b, T), b)
    r = α₀ == 0 ? r : axpy!(-α₀, x₀, r)
    r = axpy!(-α₁, y₀, r)
    x = copyto!(similar(r), x₀)
    normr = norm(r)
    S = typeof(normr)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol::S = alg.tol
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    normr < tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    # First iteration
    ρ = normr^2
    p = copyto!(similar(r), r)
    q = apply(operator, p)
    if α₀ != zero(α₀) || α₁ != one(α₁)
        axpby!(α₀, p, α₁, q)
    end
    α = ρ / dot(p,q)
    axpy!(+α, p, x)
    axpy!(-α, q, r)
    normr = norm(r)
    ρold = ρ
    ρ = normr^2
    β = ρ/ρold
    numops += 1
    numiter += 1
    if alg.verbosity > 1
        msg = "CG linsolve in iter $numiter: "
        msg *= "normres = "
        msg *= @sprintf("%.12e", normr)
        @info msg
    end

    # Check for early return
    normr < tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while numiter < maxiter
        axpby!(1, r, β, p)
        q = apply(operator, p)
        if α₀ != zero(α₀) || α₁ != one(α₁)
            axpby!(α₀, p, α₁, q)
        end
        α = ρ / dot(p,q)
        axpy!(+α, p, x)
        axpy!(-α, q, r)
        normr = norm(r)
        if normr < tol # recompute to account for buildup of floating point errors
            r = copyto!(similar(b, T), b)
            r = α₀ == 0 ? r : axpy!(-α₀, x, r)
            r = axpy!(-α₁, apply(operator, x), r)
            normr = norm(r)
            ρ = normr^2
            β = 0 # restart CG
        else
            ρold = ρ
            ρ = normr^2
            β = ρ/ρold
        end
        if normr < tol
            if alg.verbosity > 0
                @info """CG linsolve converged at iteration $numiter:
                 *  norm of residual = $normr
                 *  number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        numops += 1
        numiter += 1
        if alg.verbosity > 1
            msg = "CG linsolve in iter $numiter: "
            msg *= "normres = "
            msg *= @sprintf("%.12e", normr)
            @info msg
        end
    end
    if alg.verbosity > 0
        @warn """CG linsolve finished without converging after $numiter iterations:
         *  norm of residual = $normr
         *  number of operations = $numops"""
    end
    return (x, ConvergenceInfo(0, r, normr, numiter, numops))
end
