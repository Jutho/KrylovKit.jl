function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Lanczos;
                  alg_rrule=Arnoldi(; tol=alg.tol,
                                    krylovdim=alg.krylovdim,
                                    maxiter=alg.maxiter,
                                    eager=alg.eager,
                                    orth=alg.orth))
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    if howmany > krylovdim
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
    end

    ## FIRST ITERATION: setting up
    # Initialize Lanczos factorization
    iter = LanczosIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity=alg.verbosity)
    numops = 1
    numiter = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::typeof(β) = alg.tol

    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim + 1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    converged = 0
    local D, U, f
    while true
        β = normres(fact)
        K = length(fact)

        # diagonalize Krylov factorization
        if β <= tol && K < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`)."
                @warn msg
            end
        end
        if K == krylovdim || β <= tol || (alg.eager && K >= howmany)
            U = copyto!(view(UU, 1:K, 1:K), I)
            f = view(HH, K + 1, 1:K)
            T = rayleighquotient(fact) # symtridiagonal

            # compute eigenvalues
            if K == 1
                D = [T[1, 1]]
                f[1] = β
                converged = Int(β <= tol)
            else
                if K < krylovdim
                    T = deepcopy(T)
                end
                D, U = tridiageigh!(T, U)
                by, rev = eigsort(which)
                p = sortperm(D; by=by, rev=rev)
                D, U = permuteeig!(D, U, p)
                mul!(f, view(U, K, :), β)
                converged = 0
                while converged < K && abs(f[converged + 1]) <= tol
                    converged += 1
                end
            end

            if converged >= howmany || β <= tol
                break
            elseif alg.verbosity >= EACHITERATION_LEVEL
                @info "Lanczos eigsolve in iteration $numiter, step = $K: $converged values converged, normres = $(normres2string(abs.(f[1:howmany])))"
            end
        end

        if K < krylovdim # expand Krylov factorization
            fact = expand!(iter, fact; verbosity=alg.verbosity)
            numops += 1
        else ## shrink and restart
            if numiter == maxiter
                break
            end

            # Determine how many to keep
            keep = div(3 * krylovdim + 2 * converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

            # Restore Lanczos form in the first keep columns
            H = fill!(view(HH, 1:(keep + 1), 1:keep), zero(eltype(HH)))
            @inbounds for j in 1:keep
                H[j, j] = D[j]
                H[keep + 1, j] = f[j]
            end
            @inbounds for j in keep:-1:1
                h, ν = householder(H, j + 1, 1:j, j)
                H[j + 1, j] = ν
                H[j + 1, 1:(j - 1)] .= zero(eltype(H))
                lmul!(h, H)
                rmul!(view(H, 1:j, :), h')
                rmul!(U, h')
            end
            @inbounds for j in 1:keep
                fact.αs[j] = H[j, j]
                fact.βs[j] = H[j + 1, j]
            end

            # Update B by applying U using Householder reflections
            B = basis(fact)
            B = basistransform!(B, view(U, :, 1:keep))
            r = residual(fact)
            B[keep + 1] = scale!!(r, 1 / β)

            # Shrink Lanczos factorization
            fact = shrink!(fact, keep; verbosity=alg.verbosity)
            numiter += 1
        end
    end

    howmany′ = howmany
    if converged > howmany
        howmany′ = converged
    elseif length(D) < howmany
        howmany′ = length(D)
    end
    values = D[1:howmany′]

    # Compute eigenvectors
    V = view(U, :, 1:howmany′)

    # Compute convergence information
    vectors = let B = basis(fact)
        [B * v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [scale(r, last(v)) for v in cols(V)]
    end
    normresiduals = let f = f
        map(i -> abs(f[i]), 1:howmany′)
    end

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """Lanczos eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """Lanczos eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end


function eigsolve(A, x₀, howmany::Int, which::Selector, alg::BlockLanczos; tol = 1e-10, maxiter = 200)
	X1 = x₀
	n = size(A, 1)
    p = alg.block_size
	if n % p != 0
		error("n must be divisible by p")
	end
	r = ceil(Int64, n / p)
	X = [X1]
	M = [(X1' * A * X1 + (X1' * A * X1)') / 2]
	AX1 = A * X1
	R1 = AX1 - X1 * M[1]
	X2, B1 = qr(R1)
	X2 = Matrix(X2)
	X2 = X2 - X1 * (X1' * X2)
	X2 = X2 ./ sqrt.(sum(abs2.(X2), dims = 1))
	B = [B1]
	M2 = X2' * A * X2
	M2 = (M2 + M2') / 2
	push!(X, X2)
	push!(M, M2)
	numiter = 1
	numops = 2  # 初始矩阵乘法次数
	for k in 2:min(maxiter, r - 1)
		Rk = A * X[k] - X[k] * M[k] - X[k-1] * B[k-1]'
		Xkplus1, Bk = qr(Rk)
		Xkplus1 = Matrix(Xkplus1)
		for Y in X
			Xkplus1 = Xkplus1 - Y * (Y' * Xkplus1)
		end
		Xkplus1 = Xkplus1 ./ sqrt.(sum(abs2.(Xkplus1), dims = 1))
		push!(X, Xkplus1)
		push!(B, Bk)
		Mkplus1 = Xkplus1' * A * Xkplus1
		Mkplus1 = (Mkplus1 + Mkplus1') / 2
		push!(M, Mkplus1)
		numops += 2  # 每次迭代的矩阵乘法次数
		if norm(Bk) < tol || k == n
			break
		end
		numiter += 1
	end
	m = length(M)
	TDB = zeros(m * p, m * p)
	for i in 1:m
		TDB[i*p-p+1:i*p, i*p-p+1:i*p] = M[i]
		if i != m
			TDB[i*p-p+1:i*p, i*p+1:(i+1)*p] = B[i]'
			TDB[i*p+1:(i+1)*p, i*p-p+1:i*p] = B[i]
		end
	end

	# diagonalize TDB and return 
	D, U = LinearAlgebra.eigen(TDB)
	by, rev = eigsort(which)
	p = sortperm(D; by = by, rev = rev)
	D, U = permuteeig!(D, U, p)
    howmany′ = min(howmany, length(D))
	values = D[1:howmany′]
    vectors = hcat(X...) * U[:, 1:howmany′]

    # 计算残差
    residuals = [A * v - λ * v for (v, λ) in zip(eachcol(vectors), values)]
    normresiduals = [norm(r) for r in residuals]
    converged = count(x -> x < tol, normresiduals)

    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """Block Lanczos eigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """Block Lanczos eigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values, vectors, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end
