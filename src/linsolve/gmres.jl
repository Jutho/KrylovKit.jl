linsolve(operator, b, alg::GMRES; kwargs...) =
    linsolve(operator, b, zero(b), alg; kwargs...)

function linsolve(operator, b, x₀, alg::GMRES; a₀ = zero(eltype(b)), a₁ = one(eltype(b)))
    # Initial function operation defines number type
    r = apply(operator,x₀)
    T = eltype(r) # number type
    α₀::T = a₀
    α₁::T = a₁
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    axpy!(+1,b,scale!(-α₁,r))
    α₀ != 0 && axpy!(-α₀, x₀, r)
    x = copy!(similar(r), x₀)
    β = vecnorm(r)
    Tr = typeof(β) # real number type for norms and tolerances

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::Tr = max(alg.tol, vecnorm(b)*alg.reltol)

    # Check for early return
    β < tol && return (x, ConvergenceInfo(1, β, r, 0, 1))

    # Initialize data structures
    y = Vector{T}(krylovdim+1)
    gs = Vector{Givens{T}}(krylovdim)
    R = zeros(T, (krylovdim,krylovdim))
    numiter = 0
    numops = 1

    iter = arnoldi(operator, r, alg.orth; krylovdim = krylovdim)
    fact = start(iter)
    while numiter < maxiter # restart loop
        numiter += 1
        k = 0
        y[k+1] = β
        while β > tol && !done(iter, fact) # inner arnoldi loop
            next!(iter, fact)
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
            β = abs(y[k+1])::Tr
        end

        # Solve upper triangular system
        V = fact.V
        # r1 = y[k+1]*(gs[k].c*V[k]-gs[k].s*V[k+1])
        utldiv!(R, y, 1:k)


        # Update x
        @inbounds for i = 1:k
            axpy!(y[i], fact.V[i], x)
        end


        # Recompute residual without reevaluating operator
        for i = 1:k
            rmulc!(V,gs[i])
        end
        scale!(r, y[k+1], V[k+1])

        # Recompute residual. Could be done without operator evaluation, but
        # here we prevent growth of small numerical errors.
        # scale!(r, -α₁, apply(operator,x))
        # axpy!(+1, b, r)
        # α₀ != 0 && axpy!(-α₀, x, r) # r = b - a₀ * x - a₁ * operator(x)
        # numops += 1

        β = vecnorm(r)
        β < tol && return (x, ConvergenceInfo(1, β, r, numiter, numops))

        # Restart Arnoldi factorization
        iter = arnoldi(operator, r, alg.orth; krylovdim = krylovdim)
        start!(iter, fact)
    end
    return (x, ConvergenceInfo(0, β, r, numiter, numops))
end
