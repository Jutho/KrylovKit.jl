function linsolve(operator, b, x₀, alg::GMRES, a₀::Number = 0, a₁::Number = 1; alg_rrule = alg)
    ndeflate = alg.ndeflate
    name = ndeflate > 0 ? "GMRES-DR" : "GMRES"

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀)
    T = typeof(inner(b, y₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)::T
    α₁ = convert(T, a₁)::T
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = scale(b, one(T))
    r = iszero(α₀) ? r : add!!(r, x₀, -α₀)
    r = add!!(r, y₀, -α₁)
    x = scale!!(zerovector(r), x₀, 1)
    β = norm(r)
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = alg.tol

    # Check for early return
    if β < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """$name linsolve converged without any iterations:
            * norm of residual = $(normres2string(β))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, β, 0, 1))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "$name linsolve starts with norm of residual = $(normres2string(β))"
    end

    # Initialize data structures
    y = Vector{T}(undef, krylovdim + 1)
    gs = Vector{Givens{T}}(undef, krylovdim)
    R = fill(zero(T), (krylovdim, krylovdim))
    numiter = 0
    numops = 1 # operator has been applied once to determine T and r

    iter = ArnoldiIterator(operator, r, alg.orth)
    fact = initialize(iter; verbosity = SILENT_LEVEL)
    sizehint!(fact, alg.krylovdim)
    numops += 1 # start applies operator once
    y[1] = β
    y[2] = zero(T)

    while true # restart loop
        numiter += 1
        # QR of the starting Hessenberg matrix; y[1:(k + 1)] holds the residual in the basis
        # extended with residual(fact) / normres(fact)
        k = length(fact)
        H = rayleighquotient(fact)
        for j in 1:k
            _gmres_hessenbergcolumn!(R, gs, H, j, α₀, α₁)
            hj = j < k ? H[j + 1, j] : normres(fact)
            gs[j], R[j, j] = givens(R[j, j], α₁ * hj, j, j + 1)
            lmul!(gs[j], y)
        end
        β = convert(S, abs(y[k + 1]))

        while (!iszero(R[k, k]) && β > tol && length(fact) < krylovdim) # inner arnoldi loop
            if alg.verbosity >= EACHITERATION_LEVEL
                @info "$name linsolve in iteration $numiter; step $k: normres = $(normres2string(β))"
            end
            fact = expand!(iter, fact; verbosity = SILENT_LEVEL)
            numops += 1 # expand! applies the operator once
            k = length(fact)
            H = rayleighquotient(fact)
            _gmres_hessenbergcolumn!(R, gs, H, k, α₀, α₁)

            # Compute new Givens rotations for final row and also apply to right hand side
            if hypot(R[k, k], α₁ * normres(fact)) < tol
                if alg.verbosity >= WARN_LEVEL
                    @warn "$name linsolve in iteration $numiter; step $k: linear operator is singular in Krylov subspace"
                end
                # rotate all the weight into y[k + 1]
                gs[k], y[k + 1] = givens(zero(T), y[k], k + 1, k)
                y[k] = zero(T)
                R[k, k] = zero(T)
            else
                gs[k], R[k, k] = givens(R[k, k], α₁ * normres(fact), k, k + 1)
                y[k + 1] = zero(T)
                lmul!(gs[k], y)
            end

            # New error
            β = convert(S, abs(y[k + 1]))
        end

        # Solve upper triangular system
        if iszero(R[k, k]) && iszero(y[k])
            ldiv!(UpperTriangular(R), y, 1:(k - 1))
        else
            ldiv!(UpperTriangular(R), y, 1:k)
        end

        # Update x
        V = basis(fact)
        @inbounds for i in 1:k
            x = add!!(x, V[i], y[i])
        end

        kept = 0 # length of the factorization kept by a deflated restart
        if β > tol && numiter < maxiter
            if ndeflate > 0 && k == krylovdim && !iszero(R[k, k])
                kept = _gmresdr_deflate!(fact, y, gs, α₀, α₁, ndeflate)
            end
            if kept == 0
                # Recompute residual without reevaluating operator
                w = residual(fact)
                push!(V, scale!!(w, 1 / normres(fact)))
                for i in 1:k
                    rmul!(V, gs[i]')
                end
                r = scale!!(r, V[k + 1], y[k + 1])
            end
        else
            # Recompute residual and its norm explicitly, to ensure that no
            # numerical errors have accumulated
            r = scale!!(r, b, 1)
            r = add!!(r, apply(operator, x, α₀, α₁), -1)
            numops += 1
            β = norm(r)
            if β < tol
                if alg.verbosity >= STARTSTOP_LEVEL
                    @info """$name linsolve converged at iteration $numiter, step $k:
                    * norm of residual = $(normres2string(β))
                    * number of operations = $numops"""
                end
                return (x, ConvergenceInfo(1, r, β, numiter, numops))
            end
        end
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """$name linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(β))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, β, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "$name linsolve in iteration $numiter; step $k: normres = $(normres2string(β))"
        end

        if kept == 0
            # Restart Arnoldi factorization with new r
            iter = ArnoldiIterator(operator, r, alg.orth)
            fact = initialize!(iter, fact; verbosity = SILENT_LEVEL)
            numops += 1 # initialize! applies operator once
            y[1] = β
            y[2] = zero(T)
        end
    end
    return
end

# Copy column k of α₀ + α₁ * H into R and apply the previous Givens rotations
function _gmres_hessenbergcolumn!(R, gs, H, k, α₀, α₁)
    @inbounds begin
        for i in 1:(k - 1)
            R[i, k] = α₁ * H[i, k]
        end
        R[k, k] = α₀ + α₁ * H[k, k]
        Rk = view(R, :, k)
        for i in 1:(k - 1)
            lmul!(gs[i], Rk)
        end
    end
    return R
end

# Deflated restart of GMRES-DR (R. B. Morgan, SIAM J. Sci. Comput. 24, 20 (2002)): shrink the
# Arnoldi factorization `fact` of length m, with Hessenberg matrix G = α₀ + α₁ * H of the shifted
# map and g = α₁ * normres(fact), to length k, in four steps:
# 1. U: orthonormal Schur basis of the `ndeflate` harmonic Ritz values of smallest modulus, the
#    eigenvalues of G + |g|² G⁻ᴴ eₘ eₘᵀ (complex conjugate pairs are kept together);
# 2. w: unit vector along [-conj(g) G⁻ᴴ eₘ; 1], which spans the orthogonal complement of the
#    range of [G; g eₘᵀ] and thus contains the GMRES residual, orthogonalized against [U; 0];
# 3. the projected Hessenberg matrix [U; 0]ᴴ [H; h eₘᵀ] U and its last row wᴴ [H; h eₘᵀ] U,
#    restored to Arnoldi form, such that A * V * U = V * U * Hk + [V v] * w * hk';
# 4. the residual coordinates, the basis [V v] * [[U; 0] w] and the factorization.
# Returns k, with the residual coordinates in y[1:(k + 1)], or 0 if this is not possible.
function _gmresdr_deflate!(fact::ArnoldiFactorization, y, gs, α₀, α₁, ndeflate)
    m = length(fact)
    T = eltype(y)
    H = Matrix{T}(rayleighquotient(fact))
    h = normres(fact)
    G = α₁ * H + α₀ * I
    g = α₁ * h

    # 1. harmonic Ritz values: eigenvalues of the Hessenberg matrix G + |g|² G⁻ᴴ eₘ eₘᵀ
    F = lu(G'; check = false)
    issuccess(F) || return 0
    f = zeros(T, m)
    f[m] = one(T)
    LinearAlgebra.ldiv!(F, f)
    M = copy(G)
    view(M, :, m) .+= abs2(g) .* f
    M, Q, vals = hschur!(M, Matrix{T}(I, m, m))
    select = fill(false, m)
    select[view(sortperm(vals; by = abs), 1:ndeflate)] .= true
    for i in 1:(m - 1) # keep 2x2 blocks of a real Schur form together
        if !iszero(M[i + 1, i]) && select[i] != select[i + 1]
            select[i] = select[i + 1] = count(select) < m - 1
        end
    end
    k = count(select)
    k == 0 && return 0
    M, Q, = partitionschur!(M, Q, select)
    U = view(Q, :, 1:k)

    # 2. residual direction
    w = vcat(-conj(g) .* f, one(T))
    wm = view(w, 1:m)
    for _ in 1:2
        wm .-= U * (U' * wm)
    end
    w ./= norm(w)

    # 3. projected Hessenberg matrix, restored to Arnoldi form
    HU = H * U
    Hk = zeros(T, k + 1, k + 1)
    Hk[1:k, 1:k] = U' * HU
    hk = HU' * wm .+ (w[m + 1] * h) .* conj.(view(U, m, :))
    _restorearnoldiform!(U, Hk, hk, k)

    # 4. residual coordinates, basis and factorization
    z = zeros(T, m + 1)
    z[m + 1] = y[m + 1]
    for i in m:-1:1
        lmul!(gs[i]', z)
    end
    y[1:k] = U' * view(z, 1:m)
    y[k + 1] = dot(w, z)

    V = basis(fact)
    push!(V, scale!!(residual(fact), 1 / h))
    basistransform!(V, hcat(vcat(U, zeros(T, 1, k)), w))
    B = rayleighquotient(fact)
    for j in 1:k, i in 1:(j + 1)
        B[i, j] = Hk[i, j]
    end
    shrink!(fact, k)
    return k
end
