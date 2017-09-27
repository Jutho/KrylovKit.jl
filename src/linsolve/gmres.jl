linsolve(operator, b, alg::GMRES, a₀ = 0, a₁ = 1) =
    linsolve(operator, b, zero(b), alg, a₀, a₁)

function linsolve(operator, b, x₀, alg::GMRES, a₀ = 0, a₁ = 1)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(vecdot(b, y₀)/vecnorm(b)*one(a₀)*one(a₁))
    α₀::T = a₀
    α₁::T = a₁
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = b - α₀*x₀ - α₁*y₀
    x = copy!(similar(r), x₀)
    β = vecnorm(r)
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = max(alg.tol, vecnorm(b)*alg.reltol)

    # Check for early return
    β < tol && return (x, ConvergenceInfo(1, β, r, 0, 1))

    # Initialize data structures
    y = Vector{T}(krylovdim+1)
    gs = Vector{Givens{T}}(krylovdim)
    R = zeros(T, (krylovdim,krylovdim))
    numiter = 0
    numops = 1 # operator has been applied once to determine T

    iter = ArnoldiIterator(operator, r, alg.orth; krylovdim = krylovdim)
    fact = start(iter)
    numops += 1 # start applies operator once
    while numiter < maxiter # restart loop
        numiter += 1
        y[1] = β
        k = 1
        H = fact.H
        R[1,1] = α₀ + α₁ * fact.H[1,1]
        gs[1], R[1,1] = givens(R[1,1], α₁*H[2,1], 1, 2)
        y[2] = zero(T)
        lmul!(y, gs[1])
        β = convert(S, abs(y[2]))
        # info("iter $numiter, step $k : normres = $β")

        while β > tol && !done(iter, fact) # inner arnoldi loop
            fact = next!(iter, fact)
            numops += 1 # next! applies the operator once
            k = fact.k
            H = fact.H

            # copy Arnoldi Hessenberg matrix into R
            @inbounds for i=1:k-1
                R[i,k] = α₁ * H[i,k]
            end
            R[k,k] = α₀ + α₁ * H[k,k]

            # Apply Givens rotations
            @inbounds for i=1:k-1
                lmul!(R, gs[i], k:k)
            end
            gs[k], R[k,k] = givens(R[k,k], α₁*H[k+1,k], k, k+1)

            # Apply Givens rotations to right hand side
            y[k+1] = zero(T)
            lmul!(y, gs[k])

            # New error
            β = convert(S, abs(y[k+1]))
            # info("iter $numiter, step $k : normres = $β")
        end

        # Solve upper triangular system
        ldiv!(UpperTriangular(R), y, 1:k)

        # Update x
        V = fact.V
        @inbounds for i = 1:k
            axpy!(y[i], fact.V[i], x)
        end

        if β > tol
            # Recompute residual without reevaluating operator
            for i = 1:k
                rmulc!(V, gs[i])
            end
            scale!(r, y[k+1], V[k+1])
        else
            # Recompute residual and its norm explicitly, to ensure that no
            # numerical errors have accumulated
            scale!(r, -α₁, apply(operator, x))
            axpy!(+1, b, r)
            α₀ != 0 && axpy!(-α₀, x, r) # r = b - a₀ * x - a₁ * operator(x)
            numops += 1
            β = vecnorm(r)
            β < tol && return (x, ConvergenceInfo(1, β, r, numiter, numops))
        end

        # Restart Arnoldi factorization with new r
        iter = ArnoldiIterator(operator, r, alg.orth; krylovdim = krylovdim)
        fact = start!(iter, fact)
    end
    return (x, ConvergenceInfo(0, β, r, numiter, numops))
end
