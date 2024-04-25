# Reverse rule adopted from tsvd! rrule as found in TensorKit.jl
function ChainRulesCore.rrule(::typeof(svdsolve), A, x₀, howmany::Int, which::Symbol,
                              alg::GKL)
    val, lvec, rvec, info = svdsolve(A, x₀, howmany, which, alg)

    function svdsolve_pullback((Δval, Δlvec, Δrvec, Δinfo))
        # TODO: These type conversion should be probably handled differently
        U = hcat(lvec...)
        S = diagm(val)
        V = copy(hcat(rvec...)')
        ΔU = Δlvec isa ZeroTangent ? Δlvec : hcat(Δlvec...)
        ΔS = Δval isa ZeroTangent ? Δval : diagm(Δval)
        ΔV = Δrvec isa ZeroTangent ? Δrvec : hcat(Δrvec...)

        ∂A = truncsvd_rrule(A, U, S, V, ΔU, ΔS, ΔV)
        return NoTangent(), ∂A, ZeroTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return (val, lvec, rvec, info), svdsolve_pullback
end

# SVD adjoint with correct truncation contribution
# as presented in: https://arxiv.org/abs/2311.11894 
function truncsvd_rrule(A,
                        U,
                        S,
                        V,
                        ΔU,
                        ΔS,
                        ΔV;
                        atol::Real=0,
                        rtol::Real=atol > 0 ? 0 : eps(scalartype(S))^(3 / 4),)
    Ad = copy(A')
    tol = atol > 0 ? atol : rtol * S[1, 1]
    S⁻¹ = pinv(S; atol=tol)

    # Compute possibly divergent F terms
    F = similar(S)
    @inbounds for i in axes(F, 1), j in axes(F, 2)
        F[i, j] = if i == j
            zero(T)
        else
            sᵢ, sⱼ = S[i, i], S[j, j]
            Δs = abs(sⱼ - sᵢ) < tol ? tol : sⱼ^2 - sᵢ^2
            1 / Δs
        end
    end

    # dS contribution
    term = ΔS isa ZeroTangent ? ΔS : Diagonal(real.(ΔS))

    # dU₁ and dV₁ off-diagonal contribution
    J = F .* (U' * ΔU)
    term += (J + J') * S
    VΔV = (V * ΔV')
    K = F .* VΔV
    term += S * (K + K')

    # dV₁ diagonal contribution (diagonal of dU₁ is gauged away)
    if scalartype(U) <: Complex && !(ΔV isa ZeroTangent) && !(ΔU isa ZeroTangent)
        L = Diagonal(VΔV)
        term += 0.5 * S⁻¹ * (L' - L)
    end
    ΔA = U * term * V

    # Projector contribution for non-square A and dU₂ and dV₂
    UUd = U * U'
    VdV = V' * V
    Uproj = one(UUd) - UUd
    Vproj = one(VdV) - VdV

    # Truncation contribution from dU₂ and dV₂
    function svdlinprob(v)  # Left-preconditioned linear problem
        Γ1 = v[1] - S⁻¹ * v[2] * Vproj * Ad
        Γ2 = v[2] - S⁻¹ * v[1] * Uproj * A
        return (Γ1, Γ2)
    end
    if ΔU isa ZeroTangent && ΔV isa ZeroTangent
        m, k, n = size(U, 1), size(U, 2), size(V, 2)
        y = (zeros(scalartype(A), k * m), zeros(scalartype(A), k * n))
        γ, = linsolve(svdlinprob, y; rtol=eps(real(scalartype(A))))
    else
        y = (S⁻¹ * ΔU' * Uproj, S⁻¹ * ΔV * Vproj)
        γ, = linsolve(svdlinprob, y; rtol=eps(real(scalartype(A))))
    end
    ΔA += Uproj * γ[1]' * V + U * γ[2] * Vproj

    return ΔA
end