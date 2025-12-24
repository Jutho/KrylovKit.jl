function ChainRulesCore.rrule(config::RuleConfig,
                              ::typeof(eigsolve),
                              f,
                              x₀,
                              howmany,
                              which,
                              alg_primal;
                              alg_rrule=Arnoldi(;
                                                tol=alg_primal.tol,
                                                krylovdim=alg_primal.krylovdim,
                                                maxiter=alg_primal.maxiter,
                                                eager=alg_primal.eager,
                                                orth=alg_primal.orth,
                                                verbosity=alg_primal.verbosity))
    (vals, vecs, info) = eigsolve(f, x₀, howmany, which, alg_primal)
    if alg_primal isa Lanczos
        fᴴ = f
    elseif f isa AbstractMatrix
        fᴴ = adjoint(f)
    else
        fᴴ = let pb = rrule_via_ad(config, f, zerovector(x₀, complex(scalartype(x₀))))[2]
            v -> unthunk(pb(v)[2])
        end
    end
    eigsolve_pullback = make_eigsolve_pullback(config, f, fᴴ, x₀, howmany, which,
                                               alg_primal, alg_rrule, vals, vecs, info)
    return (vals, vecs, info), eigsolve_pullback
end

function make_eigsolve_pullback(config, f, fᴴ, x₀, howmany, which, alg_primal, alg_rrule,
                                vals, vecs, info)
    function eigsolve_pullback(ΔX)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂howmany = NoTangent()
        ∂which = NoTangent()
        ∂alg = NoTangent()

        # Prepare inputs:
        #----------------
        _Δvals = unthunk(ΔX[1])
        _Δvecs = unthunk(ΔX[2])
        # special case: propagate zero tangent
        if _Δvals isa AbstractZero && _Δvecs isa AbstractZero
            ∂f = ZeroTangent()
            return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
        end
        # discard vals/vecs from n + 1 onwards if contribution is zero
        _n_vals = _Δvals isa AbstractZero ? nothing : findlast(!iszero, _Δvals)
        _n_vecs = _Δvecs isa AbstractZero ? nothing :
                  findlast(!Base.Fix2(isa, AbstractZero), _Δvecs)
        n_vals = isnothing(_n_vals) ? 0 : _n_vals
        n_vecs = isnothing(_n_vecs) ? 0 : _n_vecs
        n = max(n_vals, n_vecs)
        # special case (can this happen?): try to maintain type stability
        if n == 0
            if howmany == 0
                T = (alg_primal isa Lanczos) ? scalartype(x₀) : complex(scalartype(x₀))
                _vecs = [zerovector(x₀, T)]
                ws = [_vecs[1]]
                ∂f = construct∂f_eig(config, f, _vecs, ws)
                return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
            else
                ws = [zerovector(vecs[1])]
                ∂f = construct∂f_eig(config, f, vecs, ws)
                return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
            end
        end
        Δvals = fill(zero(vals[1]), n)
        if n_vals > 0
            Δvals[1:n_vals] .= view(_Δvals, 1:n_vals)
        end
        if _Δvecs isa AbstractZero
            # case of no contribution of singular vectors
            Δvecs = fill(ZeroTangent(), n)
        else
            Δvecs = fill(zerovector(vecs[1]), n)
            if n_vecs > 0
                for i in 1:n_vecs
                    if !(_Δvecs[i] isa AbstractZero)
                        Δvecs[i] = _Δvecs[i]
                    end
                end
            end
        end

        # Compute actual pullback data:
        #------------------------------
        ws = compute_eigsolve_pullback_data(Δvals, Δvecs, view(vals, 1:n), view(vecs, 1:n),
                                            info, which, fᴴ, alg_primal, alg_rrule)

        # Return pullback in correct form:
        #---------------------------------
        ∂f = construct∂f_eig(config, f, vecs, ws)
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return eigsolve_pullback
end

function compute_eigsolve_pullback_data(Δvals, Δvecs, vals, vecs, info, which, fᴴ,
                                        alg_primal, alg_rrule::Union{GMRES,BiCGStab})
    ws = similar(vecs, length(Δvecs))
    T = scalartype(vecs[1])
    @inbounds for i in 1:length(Δvecs)
        Δλ = Δvals[i]
        Δv = Δvecs[i]
        λ = vals[i]
        v = vecs[i]

        # First treat special cases
        if isa(Δv, AbstractZero) && iszero(Δλ) # no contribution
            ws[i] = zerovector(v)
            continue
        end
        if isa(Δv, AbstractZero) && isa(alg_primal, Lanczos) # simple contribution
            ws[i] = scale(v, Δλ)
            continue
        end

        # TODO: Is the following useful and correct?
        # (given that Δvecs might contain weird tangent types)
        # The following only holds if `f` represents a real linear operator, which we cannot
        # check explicitly, unless `f isa AbstractMatrix`.`
        # However, exact equality between conjugate pairs of eigenvalues and eigenvectors
        # seems sufficient to guarantee this
        # Also, we can only be sure to know how to apply complex conjugation when the
        # vectors are of type `AbstractArray{T}` with `T` the scalar type
        # if i > 1 && ws[i - 1] isa AbstractArray{T} &&
        #    vals[i] == conj(vals[i - 1]) && Δvals[i] == conj(Δvals[i - 1]) &&
        #    vecs[i] == conj(vecs[i - 1]) && Δvecs[i] == conj(Δvecs[i - 1])
        #     ws[i] = conj(ws[i - 1])
        #     continue
        # end

        if isa(Δv, AbstractZero)
            b = (zerovector(v), convert(T, Δλ))
        else
            vdΔv = inner(v, Δv)
            if alg_rrule.verbosity >= WARN_LEVEL
                gauge = abs(imag(vdΔv))
                gauge > alg_primal.tol &&
                    @warn "`eigsolve` cotangent for eigenvector $i is sensitive to gauge choice: (|gauge| = $gauge)"
            end
            Δv = add(Δv, v, -vdΔv)
            b = (Δv, convert(T, Δλ))
        end
        w, reverse_info = let λ = λ, v = v
            linsolve(b, zerovector(b), alg_rrule) do (x1, x2)
                y1 = VectorInterface.add!!(VectorInterface.add!!(KrylovKit.apply(fᴴ, x1),
                                                                 x1, conj(λ), -1),
                                           v, x2)
                y2 = inner(v, x1)
                return (y1, y2)
            end
        end
        if info.converged >= i && reverse_info.converged == 0 &&
           alg_primal.verbosity >= WARN_LEVEL
            @warn "`eigsolve` cotangent linear problem ($i) did not converge, whereas the primal eigenvalue problem did: normres = $(reverse_info.normres)"
        elseif abs(w[2]) > (alg_rrule.tol * norm(w[1])) &&
               alg_primal.verbosity >= WARN_LEVEL
            @warn "`eigsolve` cotangent linear problem ($i) returns unexpected result: error = $(w[2])"
        end
        ws[i] = w[1]
    end
    return ws
end

function compute_eigsolve_pullback_data(Δvals, Δvecs, vals, vecs, info, which, fᴴ,
                                        alg_primal::Arnoldi, alg_rrule::Arnoldi)
    n = length(Δvecs)
    T = scalartype(vecs[1])
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
    if alg_rrule.verbosity >= WARN_LEVEL
        mask = abs.(transpose(vals) .- vals) .< tol
        gaugepart = VdΔV[mask] - Diagonal(real(diag(VdΔV)))[mask]
        Δgauge = norm(gaugepart, Inf)
        Δgauge > tol &&
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

    zs = similar(vecs)
    for i in 1:n
        z = scale(vecs[1], iGaVdΔV[1, i])
        for j in 2:n
            z = VectorInterface.add!!(z, vecs[j], iGaVdΔV[j, i])
        end
        zs[i] = z
    end

    # components in orthogonal subspace:
    # solve Sylvester problem (A * (1-P) + shift * P) * W - W * Λ  = ΔV as eigenvalue problem
    # with ΔVᵢ = fᴴ(zᵢ) + (1 - P) * Δvᵢ
    # where we can recylce information in the computation of P * Δvᵢ
    sylvesterarg = similar(vecs)
    for i in 1:n
        y = KrylovKit.apply(fᴴ, zs[i])
        if !(Δvecs[i] isa AbstractZero)
            y = VectorInterface.add!!(y, Δvecs[i])
            for j in 1:n
                y = VectorInterface.add!!(y, vecs[j], -iGVdΔV[j, i])
            end
        end
        sylvesterarg[i] = y
    end

    # To solve Sylvester problem as eigenvalue problem, we potentially need to shift the
    # eigenvalues zero that originate from the projection onto the orthognal complement of
    # original subspace, namely whenever zero is more extremal than the actual eigenvalues.
    # Hereto, we shift the zero eigenvalues in the original subspace to the value 2 * vals[n],
    # where we expect that if `by(vals[n]) > by(0)`, then `by(2*vals[n]) > by(vals[n])`
    # (whenever `rev = false`, and with opposite inequality whenever `rev = true`)
    by, rev = KrylovKit.eigsort(which)
    if (rev ? (by(vals[n]) < by(zero(vals[n]))) : (by(vals[n]) > by(zero(vals[n]))))
        shift = 2 * conj(vals[n])
    else
        shift = zero(vals[n])
    end
    # The ith column wᵢ of the solution to the Sylvester equation is contained in the
    # the eigenvector (wᵢ, eᵢ) corresponding to eigenvalue λᵢ of the block matrix
    # [(A * (1-P) + shift * P)  -ΔV; 0 Λ], where eᵢ is the ith unit vector. We will need
    # to renormalise the eigenvectors to have exactly eᵢ as second component. We use
    # (0, e₁ + e₂ + ... + eₙ) as the initial guess for the eigenvalue problem.

    W₀ = (zerovector(vecs[1]), one.(vals))
    P = orthogonalprojector(vecs, n, Gc)
    # TODO: is `realeigsolve` every used here, as there is a separate `alg_primal::Lanczos` method below
    solver = (T <: Real) ? KrylovKit.realeigsolve : KrylovKit.eigsolve # for `eigsolve`, `T` will always be a Complex subtype`
    rvals, Ws, reverse_info = let P = P, ΔV = sylvesterarg, shift = shift,
        eigsort = EigSorter(v -> minimum(DistanceTo(conj(v)), vals))

        solver(W₀, n, eigsort, alg_rrule) do (w, x)
            w₀ = P(w)
            w′ = KrylovKit.apply(fᴴ, add(w, w₀, -1))
            if !iszero(shift)
                w′ = VectorInterface.add!!(w′, w₀, shift)
            end
            @inbounds for i in eachindex(x) # length(x) = n but let us not use outer variables
                w′ = VectorInterface.add!!(w′, ΔV[i], -x[i])
            end
            return (w′, conj.(vals) .* x)
        end
    end
    if info.converged >= n && reverse_info.converged < n &&
       alg_primal.verbosity >= WARN_LEVEL
        @warn "`eigsolve` cotangent problem did not converge, whereas the primal eigenvalue problem did"
    end

    # cleanup and construct final result
    tol = alg_rrule.tol
    Z = zeros(T, n, n)
    for i in 1:n
        copy!(view(Z, :, i), Ws[i][2])
    end
    Zinv = inv(Z)
    error = norm(Diagonal(view(vals, 1:n))' - Z * Diagonal(view(rvals, 1:n)) * Zinv, Inf)
    if error > 10 * tol && alg_primal.verbosity >= WARN_LEVEL
        @warn "`eigsolve` cotangent linear problem returns unexpected result: error = $error vs tol = $tol"
    end
    Q = orthogonalcomplementprojector(vecs, n, Gc)
    xs = Q.(getindex.(view(Ws, 1:n), 1))
    ws = zs
    for i in 1:n
        for j in 1:n
            ws[i] = VectorInterface.add!!(ws[i], xs[j], -Zinv[j, i])
        end
    end
    return ws
end

struct DistanceTo{T}
    x::T
end
(d::DistanceTo)(y) = norm(y - d.x)

# several simplications happen in the case of a Hermitian eigenvalue problem
function compute_eigsolve_pullback_data(Δvals, Δvecs, vals, vecs, info, which, fᴴ,
                                        alg_primal::Lanczos, alg_rrule::Arnoldi)
    n = length(Δvecs)
    T = scalartype(vecs[1])
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
    if alg_rrule.verbosity >= WARN_LEVEL
        mask = abs.(transpose(vals) .- vals) .< tol
        gaugepart = view(aVdΔV, mask)
        gauge = norm(gaugepart, Inf)
        gauge > tol &&
            @warn "`eigsolve` cotangents sensitive to gauge choice: (|gauge| = $gauge)"
    end
    aVdΔV .= aVdΔV .* safe_inv.(transpose(vals) .- vals, tol)
    for i in 1:n
        aVdΔV[i, i] += real(Δvals[i])
    end

    zs = similar(vecs)
    for i in 1:n
        z = scale(vecs[1], aVdΔV[1, i])
        for j in 2:n
            z = VectorInterface.add!!(z, vecs[j], aVdΔV[j, i])
        end
        zs[i] = z
    end

    # components in orthogonal subspace
    sylvesterarg = similar(vecs)
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

    by, rev = KrylovKit.eigsort(which)
    if (rev ? (by(vals[n]) < by(zero(vals[n]))) : (by(vals[n]) > by(zero(vals[n]))))
        shift = 2 * conj(vals[n])
    else
        shift = zero(vals[n])
    end
    W₀ = (zerovector(vecs[1]), one.(vals))
    P = orthogonalprojector(vecs, n)
    solver = (T <: Real) ? KrylovKit.realeigsolve : KrylovKit.eigsolve
    rvals, Ws, reverse_info = let P = P, ΔV = sylvesterarg, shift = shift,
        eigsort = EigSorter(v -> minimum(DistanceTo(conj(v)), vals))

        solver(W₀, n, eigsort, alg_rrule) do (w, x)
            w₀ = P(w)
            w′ = KrylovKit.apply(fᴴ, add(w, w₀, -1))
            if !iszero(shift)
                w′ = VectorInterface.add!!(w′, w₀, shift)
            end
            @inbounds for i in 1:length(x) # length(x) = n but let us not use outer variables
                w′ = VectorInterface.add!!(w′, ΔV[i], -x[i])
            end
            return (w′, vals .* x)
        end
    end
    if info.converged >= n && reverse_info.converged < n &&
       alg_primal.verbosity >= WARN_LEVEL
        @warn "`eigsolve` cotangent problem did not converge, whereas the primal eigenvalue problem did"
    end

    # cleanup and construct final result
    tol = alg_rrule.tol
    Z = zeros(T, n, n)
    for i in 1:n
        copy!(view(Z, :, i), Ws[i][2])
    end
    Zinv = inv(Z)
    error = norm(Diagonal(view(vals, 1:n))' - Z * Diagonal(view(rvals, 1:n)) * Zinv, Inf)
    if error > 10 * tol && alg_primal.verbosity >= WARN_LEVEL
        @warn "`eigsolve` cotangent linear problem returns unexpected result: error = $error vs tol = $tol"
    end
    Q = orthogonalcomplementprojector(vecs, n)
    xs = Q.(getindex.(view(Ws, 1:n), 1))
    ws = zs
    for i in 1:n
        for j in 1:n
            ws[i] = VectorInterface.add!!(ws[i], xs[j], -Zinv[j, i])
        end
    end
    return ws
end

function construct∂f_eig(config, f, vecs, ws)
    config isa RuleConfig{>:HasReverseMode} ||
        throw(ArgumentError("`eigsolve` reverse-mode AD requires AD engine that supports calling back into AD"))

    v = vecs[1]
    w = ws[1]
    ∂f = rrule_via_ad(config, f, v)[2](w)[1]
    for i in 2:length(ws)
        v = vecs[i]
        w = ws[i]
        ∂f = ChainRulesCore.add!!(∂f, rrule_via_ad(config, f, v)[2](w)[1])
    end
    return ∂f
end
function construct∂f_eig(config, A::AbstractMatrix, vecs, ws)
    if A isa StridedMatrix
        return InplaceableThunk(Ā -> _buildĀ_eig!(Ā, vecs, ws),
                                @thunk(_buildĀ_eig!(zero(A), vecs, ws)))
    else
        return @thunk(ProjectTo(A)(_buildĀ_eig!(zero(A), vecs, ws)))
    end
end

function _buildĀ_eig!(Ā, vs, ws)
    for i in 1:length(ws)
        w = ws[i]
        v = vs[i]
        if !(w isa AbstractZero)
            if eltype(Ā) <: Real && eltype(w) <: Complex
                mul!(Ā, _realview(w), _realview(v)', +1, +1)
                mul!(Ā, _imagview(w), _imagview(v)', +1, +1)
            else
                mul!(Ā, w, v', +1, 1)
            end
        end
    end
    return Ā
end

function reverse_which(which)
    by, rev = KrylovKit.eigsort(which)
    return EigSorter(by ∘ conj, rev)
end
