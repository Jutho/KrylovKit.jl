function linsolve(operator, b, x₀, alg::BiCGStab, a₀::Number=0, a₁::Number=1)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(dot(b, y₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = one(T)*b # r = mul!(similar(b, T), b, 1)
    r = iszero(α₀) ? r : axpy!(-α₀, x₀, r)
    r = axpy!(-α₁, y₀, r)
    x = mul!(similar(r), x₀, 1)
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
    r_shadow = mul!(similar(r), r, 1)     # shadow residual
    ρ = dot(r_shadow, r)
    
    # Method fails if ρ is zero.
    if ρ ≈ 0.0
        @warn """BiCGStab linsolve errored after $numiter iterations:
        *   norm of residual = $normr
        *   number of operations = $numops"""
        return (x, ConvergenceInfo(0, r, normr, numiter, numops))
    end
    
    ## BiCG part of the algorithm.
    p = mul!(similar(r), r, 1)
    v = do_apply!(operator, α₀, α₁, p)
    numops += 1
    
    σ = dot(r_shadow, v)
    α = ρ / σ
    
    s = mul!(similar(r), r, 1)
    axpy!(-α, v, s)  # half step residual
    
    xhalf = mul!(similar(x), x, 1)
    axpy!(+α, p, xhalf) # half step iteration
    
    normr = norm(s)
    
    # Check for early return at half step.
    if normr < tol
        # Replace approximate residual with the actual residual.
        s = mul!(similar(b), b, 1)
        axpy!(-1, do_apply!(operator, α₀, α₁, xhalf), s) 
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
    t = do_apply!(operator, α₀, α₁, s)
    numops += 1
    
    ω = dot(t, s) / dot(t, t)
    
    mul!(x, xhalf, 1)
    axpy!(+ω, s, x)    # full step iteration
    
    mul!(r, s, 1)
    axpy!(-ω, t, r)      # full step residual
    
    # Check for early return at full step.
    normr = norm(r)
    if normr < tol
        # Replace approximate residual with the actual residual.
        mul!(r, b, 1)
        axpy!(-1, do_apply!(operator, α₀, α₁, x), r)
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
        ρ = dot(r_shadow, r)
        β = (ρ / ρold) * (α / ω)
        
        axpy!(-ω, v, p)
        axpby!(1, r, β, p)
        
        v = do_apply!(operator, α₀, α₁, p)
        numops += 1
        
        σ = dot(r_shadow, v)
        α = ρ / σ
        
        s = mul!(s, r, 1)
        s = axpy!(-α, v, s)     # half step residual
        
        xhalf = mul!(xhalf, x, 1)
        xhalf = axpy!(+α, p, xhalf)     # half step iteration
        
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
            #= s = mul!(similar(b), b, 1)
            s = iszero(α₀) ? s : axpy!(-α₀, xhalf, s)
            axpy!(-α₁, apply(operator, xhalf), s)
            numops += 1 =#
            s = mul!(similar(b), b, 1)
            axpy!(-1, do_apply!(operator, α₀, α₁, xhalf), s) 
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
        t = do_apply!(operator, α₀, α₁, s)
        numops += 1
        
        ω = dot(t, s) / dot(t, t)
        
        mul!(x, xhalf, 1)
        axpy!(+ω, s, x)    # full step iteration
        
        mul!(r, s, 1)
        axpy!(-ω, t, r)    # full step residual
        
        # Check for return at full step.
        normr = norm(r)
        if normr < tol
            # Replace approximate residual with the actual residual.
            mul!(r, b, 1)
            axpy!(-1, do_apply!(operator, α₀, α₁, x), r)
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

function do_apply!(operator, α₀, α₁, x)
    y = apply(operator, x)
    if α₀ != zero(α₀) || α₁ != one(α₁)
        axpby!(α₀, x, α₁, y)
    end

    return y
end
