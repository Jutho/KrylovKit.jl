# Reverse rule adopted from tsvd! rrule as found in TensorKit.jl
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(svdsolve), f, x₀, howmany, which,
                              alg_primal::GKL;
                              alg_rrule=Arnoldi(; tol=alg_primal.tol,
                                                krylovdim=alg_primal.krylovdim,
                                                maxiter=alg_primal.maxiter,
                                                eager=alg_primal.eager,
                                                orth=alg_primal.orth,
                                                verbosity=alg_primal.verbosity))
    vals, lvecs, rvecs, info = svdsolve(f, x₀, howmany, which, alg_primal)
    function svdsolve_pullback(ΔX)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂howmany = NoTangent()
        ∂which = NoTangent()
        ∂alg = NoTangent()

        _Δvals = unthunk(ΔX[1])
        _Δlvecs = unthunk(ΔX[2])
        _Δrvecs = unthunk(ΔX[3])

        n = 0
        while true
            if !(_Δvals isa AbstractZero) &&
               any(!iszero, view(_Δvals, (n + 1):length(_Δvals)))
                n = n + 1
                continue
            end
            if !(_Δlvecs isa AbstractZero) &&
               any(!Base.Fix2(isa, AbstractZero), view(_Δlvecs, (n + 1):length(_Δlvecs)))
                n = n + 1
                continue
            end
            if !(_Δrvecs isa AbstractZero) &&
               any(!Base.Fix2(isa, AbstractZero), view(_Δrvecs, (n + 1):length(_Δrvecs)))
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
        if _Δlvecs isa AbstractZero && _Δrvecs isa AbstractZero
            Δlvecs = fill(ZeroTangent(), n)
            Δrvecs = fill(ZeroTangent(), n)
        end
        if _Δlvecs isa AbstractZero
            Δlvecs = fill(zerovector(lvecs[1]), n)
        else
            @assert length(_Δlvecs) >= n
            Δlvecs = view(_Δlvecs, 1:n)
        end
        if _Δrvecs isa AbstractZero
            Δrvecs = fill(zerovector(rvecs[1]), n)
        else
            @assert length(_Δrvecs) >= n
            Δrvecs = view(_Δrvecs, 1:n)
        end

        xs, ys = compute_svdsolve_pullback_data(Δvals, Δlvecs, Δrvecs, view(vals, 1:n),
                                                view(lvecs, 1:n), view(rvecs, 1:n),
                                                info, f, which, alg_primal, alg_rrule)

        ∂f = construct∂f_svd(config, f, lvecs, rvecs, xs, ys)

        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, lvecs, rvecs, info), svdsolve_pullback
end

function compute_svdsolve_pullback_data(Δvals, Δlvecs, Δrvecs, vals, lvecs, rvecs, info, f,
                                        which, alg_primal, alg_rrule::Union{GMRES,BiCGStab})
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
            gaugeᵢ = abs(imag(udΔu + vdΔv))
            if gaugeᵢ > alg_primal.tol && alg_rrule.verbosity >= 1
                @warn "`svdsolve` cotangents for singular vectors $i are sensitive to gauge choice: (|gaugeᵢ| = $gaugeᵢ)"
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
        if info.converged >= i && reverse_info.converged == 0 && alg_rrule.verbosity >= 0
            @warn "`svdsolve` cotangent linear problem ($i) did not converge, whereas the primal eigenvalue problem did: normres = $(reverse_info.normres)"
        end
        x = VectorInterface.add!!(x, u, Δs / 2)
        y = VectorInterface.add!!(y, v, conj(Δs) / 2)
        xs[i] = x
        ys[i] = y
    end
    return xs, ys
end
function compute_svdsolve_pullback_data(Δvals, Δlvecs, Δrvecs, vals, lvecs, rvecs, info, f,
                                        which, alg_primal, alg_rrule::Arnoldi)
    @assert which == :LR "pullback currently only implemented for `which == :LR`
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
    mask = abs.(vals' .- vals) .< tol
    gaugepart = view(aUdΔU, mask) + view(aVdΔV, mask)
    gaugeerr = norm(gaugepart, Inf)
    if gaugeerr > alg_primal.tol && alg_rrule.verbosity >= 1
        @warn "`svdsolve` cotangents for singular vectors are sensitive to gauge choice: (|gauge| = $gaugeerr)"
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
    rvals, Ws, reverse_info = let QU = QU, QV = QV, ΔU = sylvesterargx, ΔV = sylvesterargy
        eigsolve(W₀, n, :LR, alg_rrule) do w
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
    if info.converged >= n && reverse_info.converged < n && alg_rrule.verbosity >= 0
        @warn "`svdsolve` cotangent problem did not converge, whereas the primal singular value problem did"
    end

    # cleanup and construct final result
    tol = alg_rrule.tol
    for i in 1:n
        x, y, z = Ws[i]
        _, ic = findmax(abs, z)
        if ic != i
            @warn "`svdsolve` cotangent linear problem ($ic) returns unexpected result"
        end
        factor = 1 / z[ic]
        z[ic] = zero(z[ic])
        error = max(norm(z, Inf), abs(rvals[i] - vals[ic]))
        if error > 5 * tol && alg_rrule.verbosity >= 0
            @warn "`svdsolve` cotangent linear problem ($ic) returns unexpected result: error = $error"
        end
        xs[ic] = VectorInterface.add!!(xs[ic], x, -factor)
        ys[ic] = VectorInterface.add!!(ys[ic], y, -factor)
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

function construct∂f_svd(config, (f, fᴴ)::Tuple{Any,Any}, lvecs, rvecs, xs, ys)
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
        return InplaceableThunk(Ā -> _buildĀ_svd!(Ā, lvecs, rvecs, xs, ys),
                                @thunk(_buildĀ_svd!(zero(A), lvecs, rvecs, xs, ys)))
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
