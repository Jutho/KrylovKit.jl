# Reverse rule adopted from tsvd! rrule as found in TensorKit.jl
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(svdsolve), f, x₀, howmany, which,
        alg_primal::GKL;
        alg_rrule = Arnoldi(;
            tol = alg_primal.tol,
            krylovdim = alg_primal.krylovdim,
            maxiter = alg_primal.maxiter,
            eager = alg_primal.eager,
            orth = alg_primal.orth,
            verbosity = alg_primal.verbosity
        )
    )
    vals, lvecs, rvecs, info = svdsolve(f, x₀, howmany, which, alg_primal)
    svdsolve_pullback = make_svdsolve_pullback(
        config, f, x₀, howmany, which, alg_primal,
        alg_rrule, vals, lvecs, rvecs, info
    )
    return (vals, lvecs, rvecs, info), svdsolve_pullback
end

function make_svdsolve_pullback(
        config, f, x₀, howmany, which, alg_primal, alg_rrule, vals,
        lvecs, rvecs, info
    )
    function svdsolve_pullback(ΔX)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂howmany = NoTangent()
        ∂which = NoTangent()
        ∂alg = NoTangent()

        # Prepare inputs:
        #----------------
        _Δvals = unthunk(ΔX[1])
        _Δlvecs = unthunk(ΔX[2])
        _Δrvecs = unthunk(ΔX[3])
        # special case: propagate zero tangent
        if _Δvals isa AbstractZero && _Δlvecs isa AbstractZero && _Δrvecs isa AbstractZero
            ∂f = ZeroTangent()
            return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
        end
        # discard vals/vecs from n + 1 onwards if contribution is zero
        _n_vals = _Δvals isa AbstractZero ? nothing : findlast(!iszero, _Δvals)
        _n_lvecs = _Δlvecs isa AbstractZero ? nothing :
            findlast(!Base.Fix2(isa, AbstractZero), _Δlvecs)
        _n_rvecs = _Δrvecs isa AbstractZero ? nothing :
            findlast(!Base.Fix2(isa, AbstractZero), _Δrvecs)
        n_vals = isnothing(_n_vals) ? 0 : _n_vals
        n_lvecs = isnothing(_n_lvecs) ? 0 : _n_lvecs
        n_rvecs = isnothing(_n_rvecs) ? 0 : _n_rvecs
        n = max(n_vals, n_lvecs, n_rvecs)
        # special case (can this happen?): try to maintain type stability
        if n == 0
            if howmany == 0
                _lvecs = [zerovector(x₀)]
                _rvecs = [apply_adjoint(f, x₀)]
                xs = [_lvecs[1]]
                ys = [_rvecs[1]]
                ∂f = construct∂f_svd(config, f, _lvecs, _rvecs, xs, ys)
                return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
            else
                xs = [zerovector(lvecs[1])]
                ys = [zerovector(rvecs[1])]
                ∂f = construct∂f_svd(config, f, lvecs, rvecs, xs, ys)
                return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
            end
        end
        Δvals = fill(zero(vals[1]), n)
        if n_vals > 0
            Δvals[1:n_vals] .= view(_Δvals, 1:n_vals)
        end
        if _Δlvecs isa AbstractZero && _Δrvecs isa AbstractZero
            # case of no contribution of singular vectors
            Δlvecs = fill(ZeroTangent(), n)
            Δrvecs = fill(ZeroTangent(), n)
        else
            Δlvecs = fill(zerovector(lvecs[1]), n)
            Δrvecs = fill(zerovector(rvecs[1]), n)
            if n_lvecs > 0
                Δlvecs[1:n_lvecs] .= view(_Δlvecs, 1:n_lvecs)
            end
            if n_rvecs > 0
                Δrvecs[1:n_rvecs] .= view(_Δrvecs, 1:n_rvecs)
            end
        end

        # Compute actual pullback data:
        #------------------------------
        xs, ys = compute_svdsolve_pullback_data(
            Δvals, Δlvecs, Δrvecs,
            view(vals, 1:n), view(lvecs, 1:n),
            view(rvecs, 1:n),
            info, f, which,
            alg_primal, alg_rrule
        )

        # Return pullback in correct form:
        #---------------------------------
        ∂f = construct∂f_svd(config, f, lvecs, rvecs, xs, ys)
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return svdsolve_pullback
end

function compute_svdsolve_pullback_data(
        Δvals, Δlvecs, Δrvecs,
        vals, lvecs, rvecs,
        info, f, which,
        alg_primal, alg_rrule::Union{GMRES, BiCGStab}
    )
    xs = similar(lvecs, length(Δvals))
    ys = similar(rvecs, length(Δvals))
    for i in 1:length(vals)
        Δσ = Δvals[i]
        Δu = Δlvecs[i]
        Δv = Δrvecs[i]
        σ = vals[i]
        u = lvecs[i]
        v = rvecs[i]

        # First treat special cases
        if isa(Δv, AbstractZero) && isa(Δu, AbstractZero) # no contribution
            xs[i] = scale(u, real(Δσ) / 2)
            ys[i] = scale(v, real(Δσ) / 2)
            continue
        end
        udΔu = inner(u, Δu)
        vdΔv = inner(v, Δv)
        if (udΔu isa Complex) || (vdΔv isa Complex)
            if alg_rrule.verbosity >= WARN_LEVEL
                gauge = abs(imag(udΔu + vdΔv))
                gauge > alg_primal.tol &&
                    @warn "`svdsolve` cotangents for singular vectors $i are sensitive to gauge choice: (|gauge| = $gauge)"
            end
            Δs = real(Δσ) + im * imag(udΔu - vdΔv) / (2 * σ)
        else
            Δs = real(Δσ)
        end
        b = (add(Δu, u, -udΔu), add(Δv, v, -vdΔv))
        (x, y), reverse_info = let σ = σ, u = u, v = v
            linsolve(b, zerovector(b), alg_rrule) do (x, y)
                x′ = VectorInterface.add!!(apply_normal(f, y), x, σ, -1)
                y′ = VectorInterface.add!!(apply_adjoint(f, x), y, σ, -1)
                x′ = VectorInterface.add!!(x′, u, -inner(u, x′))
                y′ = VectorInterface.add!!(y′, v, -inner(v, y′))
                return (x′, y′)
            end
        end
        if info.converged >= i && reverse_info.converged == 0 &&
                alg_primal.verbosity >= WARN_LEVEL
            @warn "`svdsolve` cotangent linear problem ($i) did not converge, whereas the primal eigenvalue problem did: normres = $(reverse_info.normres)"
        end
        x = VectorInterface.add!!(x, u, Δs / 2)
        y = VectorInterface.add!!(y, v, conj(Δs) / 2)
        xs[i] = x
        ys[i] = y
    end
    return xs, ys
end
function compute_svdsolve_pullback_data(
        Δvals, Δlvecs, Δrvecs,
        vals, lvecs, rvecs,
        info, f, which,
        alg_primal, alg_rrule::Arnoldi
    )
    @assert which == :LR "pullback currently only implemented for `which == :LR`"
    T = scalartype(lvecs)
    n = length(Δvals)
    UdΔU = zeros(T, n, n)
    VdΔV = zeros(T, n, n)
    for j in 1:n
        for i in 1:n
            if !(Δlvecs[j] isa AbstractZero)
                UdΔU[i, j] = inner(lvecs[i], Δlvecs[j])
            end
            if !(Δrvecs[j] isa AbstractZero)
                VdΔV[i, j] = inner(rvecs[i], Δrvecs[j])
            end
        end
    end
    aUdΔU = rmul!(UdΔU - UdΔU', 1 / 2)
    aVdΔV = rmul!(VdΔV - VdΔV', 1 / 2)

    tol = alg_primal.tol
    if alg_rrule.verbosity >= WARN_LEVEL
        mask = abs.(vals' .- vals) .< tol
        gaugepart = view(aUdΔU, mask) + view(aVdΔV, mask)
        gauge = norm(gaugepart, Inf)
        gauge > alg_primal.tol &&
            @warn "`svdsolve` cotangents for singular vectors are sensitive to gauge choice: (|gauge| = $gauge)"
    end
    UdΔAV = (aUdΔU .+ aVdΔV) .* safe_inv.(vals' .- vals, tol) .+
        (aUdΔU .- aVdΔV) .* safe_inv.(vals' .+ vals, tol)
    if !(Δvals isa ZeroTangent)
        UdΔAV[diagind(UdΔAV)] .+= real.(Δvals)
    end

    xs = similar(lvecs, n)
    ys = similar(rvecs, n)
    for i in 1:n
        x = scale(lvecs[1], UdΔAV[1, i] / 2)
        y = scale(rvecs[1], conj(UdΔAV[i, 1]) / 2)
        for j in 2:n
            x = VectorInterface.add!!(x, lvecs[j], UdΔAV[j, i] / 2)
            y = VectorInterface.add!!(y, rvecs[j], conj(UdΔAV[i, j]) / 2)
        end
        xs[i] = x
        ys[i] = y
    end

    sylvesterargx = similar(lvecs)
    for i in 1:n
        x = zerovector(lvecs[1])
        if !(Δlvecs[i] isa AbstractZero)
            x = VectorInterface.add!!(x, Δlvecs[i], +1)
            for j in 1:n
                x = VectorInterface.add!!(x, lvecs[j], -UdΔU[j, i])
            end
        end
        sylvesterargx[i] = x
    end
    sylvesterargy = similar(rvecs)
    for i in 1:n
        y = zerovector(rvecs[1])
        if !(Δrvecs[i] isa AbstractZero)
            y = VectorInterface.add!!(y, Δrvecs[i], +1)
            for j in 1:n
                y = VectorInterface.add!!(y, rvecs[j], -VdΔV[j, i])
            end
        end
        sylvesterargy[i] = y
    end

    W₀ = (zerovector(lvecs[1]), zerovector(rvecs[1]), fill(one(T), n))
    QU = orthogonalcomplementprojector(lvecs, n)
    QV = orthogonalcomplementprojector(rvecs, n)
    solver = (T <: Real) ? KrylovKit.realeigsolve : KrylovKit.eigsolve
    rvals, Ws, reverse_info = let QU = QU, QV = QV, ΔU = sylvesterargx, ΔV = sylvesterargy
        solver(W₀, n, :LR, alg_rrule) do w
            x, y, z = w
            x′ = QU(apply_normal(f, y))
            y′ = QV(apply_adjoint(f, x))
            @inbounds for i in 1:length(z)
                x′ = VectorInterface.add!!(x′, ΔU[i], -z[i])
                y′ = VectorInterface.add!!(y′, ΔV[i], -z[i])
            end
            return (x′, y′, vals .* z)
        end
    end
    if info.converged >= n && reverse_info.converged < n &&
            alg_primal.verbosity >= WARN_LEVEL
        @warn "`svdsolve` cotangent problem did not converge, whereas the primal singular value problem did"
    end

    # cleanup and construct final result
    tol = alg_rrule.tol
    Z = zeros(T, n, n)
    for i in 1:n
        copy!(view(Z, :, i), Ws[i][3])
    end
    Zinv = inv(Z)
    error = norm(Diagonal(view(vals, 1:n)) - Z * Diagonal(view(rvals, 1:n)) * Zinv, Inf)
    if error > 10 * tol && alg_primal.verbosity >= WARN_LEVEL
        @warn "`svdsolve` cotangent linear problem returns unexpected result: error = $error vs tol = $tol"
    end
    for i in 1:n
        for j in 1:n
            xs[i] = VectorInterface.add!!(xs[i], Ws[j][1], -Zinv[j, i])
            ys[i] = VectorInterface.add!!(ys[i], Ws[j][2], -Zinv[j, i])
        end
    end
    return xs, ys
end

function construct∂f_svd(config, f, lvecs, rvecs, xs, ys)
    config isa RuleConfig{>:HasReverseMode} ||
        throw(ArgumentError("`svdsolve` reverse-mode AD requires AD engine that supports calling back into AD"))

    u, v = lvecs[1], rvecs[1]
    x, y = xs[1], ys[1]
    ∂f = rrule_via_ad(config, f, v, Val(false))[2](x)[1]
    ∂f = ChainRulesCore.add!!(∂f, rrule_via_ad(config, f, u, Val(true))[2](y)[1])
    for i in 2:length(xs)
        u, v = lvecs[i], rvecs[i]
        x, y = xs[i], ys[i]
        ∂f = ChainRulesCore.add!!(∂f, rrule_via_ad(config, f, v, Val(false))[2](x)[1])
        ∂f = ChainRulesCore.add!!(∂f, rrule_via_ad(config, f, u, Val(true))[2](y)[1])
    end
    return ∂f
end
function construct∂f_svd(config, (f, fᴴ)::Tuple{Any, Any}, lvecs, rvecs, xs, ys)
    config isa RuleConfig{>:HasReverseMode} ||
        throw(ArgumentError("`svdsolve` reverse-mode AD requires AD engine that supports calling back into AD"))

    u, v = lvecs[1], rvecs[1]
    x, y = xs[1], ys[1]
    ∂f = rrule_via_ad(config, f, v)[2](x)[1]
    ∂fᴴ = rrule_via_ad(config, fᴴ, u)[2](y)[1]
    for i in 2:length(xs)
        u, v = lvecs[i], rvecs[i]
        x, y = xs[i], ys[i]
        ∂f = ChainRulesCore.add!!(∂f, rrule_via_ad(config, f, v)[2](x)[1])
        ∂fᴴ = ChainRulesCore.add!!(∂fᴴ, rrule_via_ad(config, fᴴ, u)[2](y)[1])
    end
    return (∂f, ∂fᴴ)
end
function construct∂f_svd(config, A::AbstractMatrix, lvecs, rvecs, xs, ys)
    if A isa StridedMatrix
        return InplaceableThunk(
            Ā -> _buildĀ_svd!(Ā, lvecs, rvecs, xs, ys),
            @thunk(_buildĀ_svd!(zero(A), lvecs, rvecs, xs, ys))
        )
    else
        return @thunk(ProjectTo(A)(_buildĀ_svd!(zero(A), lvecs, rvecs, xs, ys)))
    end
end
function _buildĀ_svd!(Ā, lvecs, rvecs, xs, ys)
    for i in 1:length(xs)
        u, v = lvecs[i], rvecs[i]
        x, y = xs[i], ys[i]
        mul!(Ā, x, v', +1, +1)
        mul!(Ā, u, y', +1, +1)
    end
    return Ā
end
