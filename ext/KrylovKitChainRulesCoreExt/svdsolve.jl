# Reverse rule adopted from tsvd! rrule as found in TensorKit.jl
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(svdsolve), f, x₀, howmany, which, alg_primal::GKL;
                              alg_rrule=GMRES(; tol=alg_primal.tol,
                                                krylovdim=alg_primal.krylovdim,
                                                maxiter=alg_primal.maxiter,
#                                                eager=alg_primal.eager,
                                                orth=alg_primal.orth))

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
        if _Δlvecs isa AbstractZero
            Δlvecs = fill(ZeroTangent(), n)
        else
            @assert length(_Δlvecs) >= n
            Δlvecs = view(_Δlvecs, 1:n)
        end
        if _Δrvecs isa AbstractZero
            Δrvecs = fill(ZeroTangent(), n)
        else
            @assert length(_Δrvecs) >= n
            Δrvecs = view(_Δrvecs, 1:n)
        end

        xs, ys  = compute_svdsolve_pullback_data(Δvals, Δlvecs, Δrvecs, view(vals, 1:n), view(lvecs, 1:n), view(rvecs, 1:n), 
                                            info, f, which, alg_primal, alg_rrule)
        ∂f = construct∂f_svd(config, f, lvecs, rvecs, xs, ys)
        
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, lvecs, rvecs, info), svdsolve_pullback
end

function compute_svdsolve_pullback_data(Δvals, Δlvecs, Δrvecs, vals, lvecs, rvecs, info, f, which, alg_primal, alg_rrule)
    xs = similar(lvecs, length(Δvals))
    ys = similar(rvecs, length(Δvals))
    for i in 1:length(vals)
        Δσ = Δvals[i]
        Δu = Δlvecs[i]
        Δv = Δrvecs[i]
        σ = vals[i]
        u = lvecs[i]
        v = rvecs[i]

        # First threat special cases
        if isa(Δv, AbstractZero) && isa(Δu, AbstractZero) # no contribution
            xs[i] = scale(u, real(Δσ)/2)
            ys[i] = scale(v, real(Δσ)/2)
            continue
        end
        udΔu = inner(u, Δu)
        vdΔv = inner(v, Δv)
        gaugeᵢ = abs(imag(udΔu + vdΔv))
        if gaugeᵢ > alg_primal.tol && alg_rrule.verbosity >= 1
            @warn "`svdsolve` cotangents for singular vectors $i are sensitive to gauge choice: (|gaugeᵢ| = $gaugeᵢ)"
        end
        

        Δs = real(Δσ) + im*imag(udΔu - vdΔv)/(2*σ)
        Δu = add(Δu, u, -udΔu)
        Δv = add(Δv, v, -vdΔv)
        b = (Δu, Δv)
        (x, y), reverse_info = let σ = σ, u = u, v = v
            linsolve(b, zerovector(b), alg_rrule) do z
                x, y = z
                x′ = VectorInterface.add!!(apply_normal(f,y), x, σ, -1)
                y′ = VectorInterface.add!!(apply_adjoint(f, x), y, σ, -1)
                x′ = VectorInterface.add!!(x′, u, -inner(u, x′))
                y′ = VectorInterface.add!!(y′, v, -inner(v, y′))
                return (x′, y′)
            end
        end
        if info.converged >= i && reverse_info.converged == 0 && alg_rrule.verbosity >= 0
            @warn "`svdsolve` cotangent linear problem ($i) did not converge, whereas the primal eigenvalue problem did: normres = $(reverse_info.normres)"
        end
        x = VectorInterface.add!!(x, u, Δs/2)
        y = VectorInterface.add!!(y, v, conj(Δs)/2)
        xs[i] = x
        ys[i] = y
    end
    return xs, ys
end

function construct∂f_svd(config, f::Any, lvecs, rvecs, xs, ys)
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

function construct∂f_svd(config, (f,fᴴ)::Tuple{Any,Any}, lvecs, rvecs, xs, ys)
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
        return InplaceableThunk(Ā -> _buildĀ_svd!(Ā, lvecs, rvecs, xs, ys), @thunk(_buildĀ_svd!(zero(A), lvecs, rvecs, xs, ys)))
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