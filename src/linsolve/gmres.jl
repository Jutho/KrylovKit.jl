linsolve(operator, b, alg::GMRES, a₀ = 0, a₁ = 1) =
    linsolve(operator, b, fill!(similar(b), zero(eltype(b))), alg, a₀, a₁)

function linsolve(operator, b, x₀, alg::GMRES, a₀ = 0, a₁ = 1)
    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(dot(b, y₀)/norm(b)*one(a₀)*one(a₁))
    α₀::T = a₀
    α₁::T = a₁
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = similar(b, T)
    copyto!(r, b)
    axpy!(-α₀, x₀, r)
    axpy!(-α₁, y₀, r)
    x = copyto!(similar(r), x₀)
    β = norm(r)
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = max(alg.tol, norm(b)*alg.reltol)

    # Check for early return
    β < tol && return (x, ConvergenceInfo(1, β, r, 0, 1))

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
        # info("iter $numiter, step $k : normres = $β")

        while β > tol && length(fact) < krylovdim # inner arnoldi loop
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
            @inbounds for i=1:k-1
                lmul!(gs[i], view(R, :, k:k))
            end
            gs[k], R[k,k] = givens(R[k,k], α₁*normres(fact), k, k+1)

            # Apply Givens rotations to right hand side
            y[k+1] = zero(T)
            lmul!(gs[k], y)

            # New error
            β = convert(S, abs(y[k+1]))
            # info("iter $numiter, step $k : normres = $β")
        end
        # info("iter $numiter, finished at step $k : normres = $β")

        # Solve upper triangular system
        ldiv!(UpperTriangular(R), y, 1:k)

        # Update x
        V = basis(fact)
        @inbounds for i = 1:k
            axpy!(y[i], V[i], x)
        end

        if β > tol
            # Recompute residual without reevaluating operator
            w = residual(fact)
            push!(V, mul!(w, w, 1/normres(fact)))
            for i = 1:k
                rmul!(V, gs[i]')
            end
            mul!(r, y[k+1], V[k+1])
        else
            # Recompute residual and its norm explicitly, to ensure that no
            # numerical errors have accumulated
            mul!(r, -α₁, apply(operator, x))
            axpy!(+1, b, r)
            α₀ != 0 && axpy!(-α₀, x, r) # r = b - a₀ * x - a₁ * operator(x)
            numops += 1
            β = norm(r)
            β < tol && return (x, ConvergenceInfo(1, β, r, numiter, numops))
        end

        # Restart Arnoldi factorization with new r
        iter = ArnoldiIterator(operator, r, alg.orth)
        fact = initialize!(iter, fact)
    end
    return (x, ConvergenceInfo(0, β, r, numiter, numops))
end
