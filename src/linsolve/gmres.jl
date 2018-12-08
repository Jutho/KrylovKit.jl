function linsolve(operator, b, x₀, alg::GMRES, a₀::Number = 0, a₁::Number = 1)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(dot(b, y₀)/norm(b)*one(a₀)*one(a₁))
    α₀ = convert(T, a₀)::T
    α₁ = convert(T, a₁)::T
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = copyto!(similar(b, T), b)
    r = α₀ == 0 ? r : axpy!(-α₀, x₀, r)
    r = axpy!(-α₁, y₀, r)
    x = copyto!(similar(r), x₀)
    β = norm(r)
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = alg.tol

    # Check for early return
    if β < tol
        if alg.verbosity > 0
            @info """GMRES linsolve converged without any iterations:
             *  norm of residual = $β
             *  number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, β, 0, 1))
    end

    # Initialize data structures
    y = Vector{T}(undef, krylovdim+1)
    gs = Vector{Givens{T}}(undef, krylovdim)
    R = fill(zero(T), (krylovdim,krylovdim))
    numiter = 0
    numops = 1 # operator has been applied once to determine T

    iter = ArnoldiIterator(operator, r, alg.orth)
    fact = initialize(iter)
    numops += 1 # start applies operator once

    while numiter < maxiter # restart loop
        numiter += 1
        y[1] = β
        k = 1
        H = rayleighquotient(fact)
        R[1,1] = α₀ + α₁ * H[1,1]
        gs[1], R[1,1] = givens(R[1,1], α₁*normres(fact), 1, 2)
        y[2] = zero(T)
        lmul!(gs[1], y)
        β = convert(S, abs(y[2]))
        if alg.verbosity > 2
            msg = "GMRES linsolve in iter $numiter; step $k: "
            msg *= "normres = "
            msg *= @sprintf("%.12e", β)
            @info msg
        end

        while (β > tol && length(fact) < krylovdim) # inner arnoldi loop
            fact = expand!(iter, fact)
            numops += 1 # expand! applies the operator once
            k = length(fact)
            H = rayleighquotient(fact)

            # copy Arnoldi Hessenberg matrix into R
            @inbounds begin
                for i=1:k-1
                    R[i,k] = α₁ * H[i,k]
                end
                R[k,k] = α₀ + α₁ * H[k,k]
            end

            # Apply Givens rotations
            Rk = view(R, :, k)
            @inbounds for i=1:k-1
                lmul!(gs[i], Rk)
            end
            gs[k], R[k,k] = givens(R[k,k], α₁*normres(fact), k, k+1)

            # Apply Givens rotations to right hand side
            y[k+1] = zero(T)
            lmul!(gs[k], y)

            # New error
            β = convert(S, abs(y[k+1]))
            if alg.verbosity > 2
                msg = "GMRES linsolve in iter $numiter; step $k: "
                msg *= "normres = "
                msg *= @sprintf("%.12e", β)
                @info msg
            end
        end
        if alg.verbosity > 1
            msg = "GMRES linsolve in iter $numiter; finised at step $k: "
            msg *= "normres = "
            msg *= @sprintf("%.12e", β)
            @info msg
        end

        # Solve upper triangular system
        y2 = copy(y)
        ldiv!(UpperTriangular(R), y, 1:k)

        # Update x
        V = basis(fact)
        @inbounds for i = 1:k
            x = axpy!(y[i], V[i], x)
        end

        if β > tol
            # Recompute residual without reevaluating operator
            w = residual(fact)
            push!(V, rmul!(w, 1/normres(fact)))
            for i = 1:k
                rmul!(V, gs[i]')
            end
            r = mul!(r, y[k+1], V[k+1])
        else
            # Recompute residual and its norm explicitly, to ensure that no
            # numerical errors have accumulated
            r = copyto!(r, b)                      # r = b
            r = α₀ == 0 ? r : axpy!(-α₀, x, r)     #      - α₀ * x
            r = axpy!(-α₁, apply(operator, x), r)  #      - α₁ * operator(x)
            numops += 1
            β = norm(r)
            if β < tol
                if alg.verbosity > 0
                    @info """GMRES linsolve converged at iteration $numiter, step $k:
                     *  norm of residual = $β
                     *  number of operations = $numops"""
                end
                return (x, ConvergenceInfo(1, r, β, numiter, numops))
            end
        end

        # Restart Arnoldi factorization with new r
        iter = ArnoldiIterator(operator, r, alg.orth)
        fact = initialize!(iter, fact)
    end

    if alg.verbosity > 0
        @warn """GMRES linsolve finished without converging after $numiter iterations:
         *  norm of residual = $β
         *  number of operations = $numops"""
    end
    return (x, ConvergenceInfo(0, r, β, numiter, numops))
end
