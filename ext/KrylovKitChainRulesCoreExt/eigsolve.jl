function ChainRulesCore.rrule(config::RuleConfig,
                              ::typeof(eigsolve),
                              f,
                              x₀,
                              howmany,
                              which,
                              alg_primal;
                              alg_rrule=Arnoldi(; tol=alg_primal.tol,
                                                krylovdim=alg_primal.krylovdim,
                                                maxiter=alg_primal.maxiter,
                                                eager=alg_primal.eager,
                                                orth=alg_primal.orth,
                                                verbosity=alg_primal.verbosity))
    (vals, vecs, info) = eigsolve(f, x₀, howmany, which, alg_primal)
    T, fᴴ, construct∂f = _prepare_inputs(config, f, vecs, alg_primal)

    function eigsolve_pullback(ΔX)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂howmany = NoTangent()
        ∂which = NoTangent()
        ∂alg = NoTangent()

        _Δvals = unthunk(ΔX[1])
        _Δvecs = unthunk(ΔX[2])

        n = 0
        while true
            if !(_Δvals isa AbstractZero) &&
               any(!iszero, view(_Δvals, (n + 1):length(_Δvals)))
                n = n + 1
                continue
            end
            if !(_Δvecs isa AbstractZero) &&
               any(!Base.Fix2(isa, AbstractZero), view(_Δvecs, (n + 1):length(_Δvecs)))
                n = n + 1
                continue
            end
            break
        end
        @assert n <= length(vals)
        if n == 0
            ∂f = ZeroTangent()
            return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
        end
        if _Δvals isa AbstractZero
            Δvals = fill(zero(vals[1]), n)
        else
            @assert length(_Δvals) >= n
            Δvals = view(_Δvals, 1:n)
        end
        if _Δvecs isa AbstractZero
            Δvecs = fill(ZeroTangent(), n)
        else
            @assert length(_Δvecs) >= n
            Δvecs = view(_Δvecs, 1:n)
        end

        ws = compute_eigsolve_pullback_data(Δvals, Δvecs, view(vals, 1:n), view(vecs, 1:n),
                                            info, which, fᴴ, T, alg_primal, alg_rrule)
        ∂f = construct∂f(ws)
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, vecs, info), eigsolve_pullback
end

function compute_eigsolve_pullback_data(Δvals, Δvecs, vals, vecs, info, which, fᴴ, T,
                                        alg_primal, alg_rrule::Union{GMRES,BiCGStab})
    ws = similar(vecs, length(Δvecs))
    @inbounds for i in 1:length(Δvecs)
        Δλ = Δvals[i]
        Δv = Δvecs[i]
        λ = vals[i]
        v = vecs[i]

        # First threat special cases
        if isa(Δv, AbstractZero) && isa(Δλ, AbstractZero) # no contribution
            ws[i] = zerovector(v)
            continue
        end
        if isa(Δv, AbstractZero) && isa(alg_primal, Lanczos) # simple contribution
            ws[i] = scale(v, Δλ)
            continue
        end

        # General case :

        # for the case where `f` is a real matrix, we can expect the following simplication
        # TODO: can we implement this within our general approach, or generalise this to also
        # cover the case where `f` is a function?
        # if i > 1 && eltype(A) <: Real &&
        #    vals[i] == conj(vals[i - 1]) && Δvals[i] == conj(Δvals[i - 1]) &&
        #    vecs[i] == conj(vecs[i - 1]) && Δvecs[i] == conj(Δvecs[i - 1])
        #     ws[i] = conj(ws[i - 1])
        #     continue
        # end

        if isa(Δv, AbstractZero)
            b = (zerovector(v), convert(T, Δλ))
        else
            vdΔv = inner(v, Δv)
            gaugeᵢ = abs(imag(vdΔv))
            if gaugeᵢ > alg_primal.tol && alg_rrule.verbosity >= 1
                @warn "`eigsolve` cotangent for eigenvector $i is sensitive to gauge choice: (|gaugeᵢ| = $gaugeᵢ)"
            end
            Δv = add(Δv, v, -vdΔv)
            b = (Δv, convert(T, Δλ))
        end
        w, reverse_info = let λ = λ, v = v
            linsolve(b, zerovector(b), alg_rrule) do x
                x1, x2 = x
                y1 = VectorInterface.add!!(VectorInterface.add!!(fᴴ(x1), x1, conj(λ), -1),
                                           v, x2)
                y2 = inner(v, x1)
                return (y1, y2)
            end
        end
        if info.converged >= i && reverse_info.converged == 0 && alg_rrule.verbosity >= 0
            @warn "`eigsolve` cotangent linear problem ($i) did not converge, whereas the primal eigenvalue problem did: normres = $(reverse_info.normres)"
        elseif abs(w[2]) > alg_rrule.tol && alg_rrule.verbosity >= 0
            @warn "`eigsolve` cotangent linear problem ($i) returns unexpected result: error = $(w[2])"
        end
        ws[i] = w[1]
    end
    return ws
end

function compute_eigsolve_pullback_data(Δvals, Δvecs, vals, vecs, info, which, fᴴ, T,
                                        alg_primal::Arnoldi, alg_rrule::Arnoldi)
    n = length(Δvecs)
    G = zeros(T, n, n)
    VdΔV = zeros(T, n, n)
    for j in 1:n
        for i in 1:n
            if i < j
                G[i, j] = conj(G[j, i])
            elseif i == j
                G[i, i] = norm(vecs[i])^2
            else
                G[i, j] = inner(vecs[i], vecs[j])
            end
            if !(Δvecs[j] isa AbstractZero)
                VdΔV[i, j] = inner(vecs[i], Δvecs[j])
            end
        end
    end

    # components along subspace spanned by current eigenvectors
    tol = alg_primal.tol
    mask = abs.(transpose(vals) .- vals) .< tol
    gaugepart = VdΔV[mask] - Diagonal(real(diag(VdΔV)))[mask]
    Δgauge = norm(gaugepart, Inf)
    if Δgauge > tol && alg_rrule.verbosity >= 1
        @warn "`eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    end
    VdΔV′ = VdΔV - G * Diagonal(diag(VdΔV) ./ diag(G))
    aVdΔV = VdΔV′ .* conj.(safe_inv.(transpose(vals) .- vals, tol))
    for i in 1:n
        aVdΔV[i, i] += Δvals[i]
    end
    Gc = cholesky!(G)
    iGaVdΔV = Gc \ aVdΔV
    iGVdΔV = Gc \ VdΔV

    zs = similar(Δvecs)
    for i in 1:n
        z = scale(vecs[1], iGaVdΔV[1, i])
        for j in 2:n
            z = VectorInterface.add!!(z, vecs[j], iGaVdΔV[j, i])
        end
        zs[i] = z
    end

    # components in orthogonal subspace
    sylvesterarg = similar(Δvecs)
    for i in 1:n
        y = fᴴ(zs[i])
        if !(Δvecs[i] isa AbstractZero)
            y = VectorInterface.add!!(y, Δvecs[i], +1)
            for j in 1:n
                y = VectorInterface.add!!(y, vecs[j], -iGVdΔV[j, i])
            end
        end
        sylvesterarg[i] = y
    end

    W₀ = (zerovector(vecs[1]), one.(vals))
    P = orthogonalprojector(vecs, n, Gc)
    by, rev = KrylovKit.eigsort(which)
    if (rev ? (by(vals[n]) < by(zero(vals[n]))) : (by(vals[n]) > by(zero(vals[n]))))
        shift = 2 * conj(vals[n])
    else
        shift = zero(vals[n])
    end
    rvals, Ws, reverse_info = let P = P, ΔV = sylvesterarg, shift = shift
        eigsolve(W₀, n, reverse_wich(which), alg_rrule) do W
            w, x = W
            w₀ = P(w)
            w′ = fᴴ(add(w, w₀, -1))
            if !iszero(shift)
                w′ = VectorInterface.add!!(w′, w₀, shift)
            end
            @inbounds for i in 1:length(x) # length(x) = n but let us not use outer variables
                w′ = VectorInterface.add!!(w′, ΔV[i], -x[i])
            end
            return (w′, conj.(vals) .* x)
        end
    end
    if info.converged >= n && reverse_info.converged < n && alg_rrule.verbosity >= 0
        @warn "`eigsolve` cotangent problem did not converge, whereas the primal eigenvalue problem did"
    end

    # cleanup and construct final result
    ws = zs
    tol = alg_rrule.tol
    Q = orthogonalcomplementprojector(vecs, n, Gc)
    for i in 1:n
        w, x = Ws[i]
        _, ic = findmax(abs, x)
        factor = 1 / x[ic]
        x[ic] = zero(x[ic])
        error = max(norm(x, Inf), abs(rvals[i] - conj(vals[ic])))
        if error > 5 * tol && alg_rrule.verbosity >= 0
            @warn "`eigsolve` cotangent linear problem ($ic) returns unexpected result: error = $error"
        end
        ws[ic] = VectorInterface.add!!(zs[ic], Q(w), -factor)
    end
    return ws
end

# several simplications happen in the case of a Hermitian eigenvalue problem
function compute_eigsolve_pullback_data(Δvals, Δvecs, vals, vecs, info, which, fᴴ, T,
                                        alg_primal::Lanczos, alg_rrule::Arnoldi)
    n = length(Δvecs)
    VdΔV = zeros(T, n, n)
    for j in 1:n
        for i in 1:n
            if !(Δvecs[j] isa AbstractZero)
                VdΔV[i, j] = inner(vecs[i], Δvecs[j])
            end
        end
    end

    # components along subspace spanned by current eigenvectors
    tol = alg_primal.tol
    aVdΔV = rmul!(VdΔV - VdΔV', 1 / 2)
    mask = abs.(transpose(vals) .- vals) .< tol
    gaugepart = view(aVdΔV, mask)
    Δgauge = norm(gaugepart, Inf)
    if Δgauge > tol && alg_rrule.verbosity >= 1
        @warn "`eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    end
    aVdΔV .= aVdΔV .* safe_inv.(transpose(vals) .- vals, tol)
    for i in 1:n
        aVdΔV[i, i] += real(Δvals[i])
    end

    zs = similar(Δvecs)
    for i in 1:n
        z = scale(vecs[1], aVdΔV[1, i])
        for j in 2:n
            z = VectorInterface.add!!(z, vecs[j], aVdΔV[j, i])
        end
        zs[i] = z
    end

    # components in orthogonal subspace
    sylvesterarg = similar(Δvecs)
    for i in 1:n
        y = zerovector(vecs[1])
        if !(Δvecs[i] isa AbstractZero)
            y = VectorInterface.add!!(y, Δvecs[i], +1)
            for j in 1:n
                y = VectorInterface.add!!(y, vecs[j], -VdΔV[j, i])
            end
        end
        sylvesterarg[i] = y
    end

    W₀ = (zerovector(vecs[1]), one.(vals))
    P = orthogonalprojector(vecs, n)
    by, rev = KrylovKit.eigsort(which)
    if (rev ? (by(vals[n]) < by(zero(vals[n]))) : (by(vals[n]) > by(zero(vals[n]))))
        shift = 2 * conj(vals[n])
    else
        shift = zero(vals[n])
    end
    rvals, Ws, reverse_info = let P = P, ΔV = sylvesterarg, shift = shift
        eigsolve(W₀, n, reverse_wich(which), alg_rrule) do W
            w, x = W
            w₀ = P(w)
            w′ = fᴴ(add(w, w₀, -1))
            if !iszero(shift)
                w′ = VectorInterface.add!!(w′, w₀, shift)
            end
            @inbounds for i in 1:length(x) # length(x) = n but let us not use outer variables
                w′ = VectorInterface.add!!(w′, ΔV[i], -x[i])
            end
            return (w′, vals .* x)
        end
    end
    if info.converged >= n && reverse_info.converged < n && alg_rrule.verbosity >= 0
        @warn "`eigsolve` cotangent problem did not converge, whereas the primal eigenvalue problem did"
    end

    # cleanup and construct final result
    ws = zs
    tol = alg_rrule.tol
    Q = orthogonalcomplementprojector(vecs, n)
    for i in 1:n
        w, x = Ws[i]
        _, ic = findmax(abs, x)
        factor = 1 / x[ic]
        x[ic] = zero(x[ic])
        error = max(norm(x, Inf), abs(rvals[i] - conj(vals[ic])))
        if error > 5 * tol && alg_rrule.verbosity >= 0
            @warn "`eigsolve` cotangent linear problem ($ic) returns unexpected result: error = $error"
        end
        ws[ic] = VectorInterface.add!!(zs[ic], Q(w), -factor)
    end
    return ws
end
