function ChainRulesCore.rrule(
        config::RuleConfig,
        ::typeof(linsolve),
        f,
        b,
        x₀,
        alg_primal,
        a₀,
        a₁; alg_rrule = alg_primal
    )
    (x, info) = linsolve(f, b, x₀, alg_primal, a₀, a₁)
    fᴴ, construct∂f = lin_preprocess(config, f, x)
    linsolve_pullback = make_linsolve_pullback(
        fᴴ, b, a₀, a₁, alg_rrule, construct∂f, x,
        info
    )
    return (x, info), linsolve_pullback
end

function make_linsolve_pullback(fᴴ, b, a₀, a₁, alg_rrule, construct∂f, x, info)
    return function linsolve_pullback(X̄)
        x̄ = unthunk(X̄[1])
        @assert X̄[2] isa AbstractZero "No cotangent of the `info` output is supported."
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂algorithm = NoTangent()
        if x̄ isa AbstractZero
            ∂f = ZeroTangent()
            ∂b = ZeroTangent()
            ∂a₀ = ZeroTangent()
            ∂a₁ = ZeroTangent()
            return ∂self, ∂f, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
        end

        x̄₀ = zerovector(
            x̄,
            VectorInterface.promote_scale(
                scalartype(x̄),
                VectorInterface.promote_scale(
                    a₀,
                    a₁
                )
            )
        )
        ∂b, reverse_info = linsolve(
            fᴴ, x̄, x̄₀, alg_rrule, conj(a₀),
            conj(a₁)
        )
        if info.converged > 0 && reverse_info.converged == 0 &&
                alg_primal.verbosity >= WARN_LEVEL
            @warn "`linsolve` cotangent problem did not converge, whereas the primal linear problem did: normres = $(reverse_info.normres)"
        end
        x∂b = inner(x, ∂b)
        b∂b = inner(b, ∂b)
        ∂f = construct∂f(scale(∂b, -conj(a₁)))
        ∂a₀ = -x∂b
        ∂a₁ = (x∂b * conj(a₀) - b∂b) / conj(a₁)

        return ∂self, ∂f, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
    end
end

function lin_preprocess(config, f, x)
    config isa RuleConfig{>:HasReverseMode} ||
        throw(ArgumentError("`linsolve` reverse-mode AD requires AD engine that supports calling back into AD"))
    pb = rrule_via_ad(config, f, x)[2]
    fᴴ, construct∂f_lin = let pb = rrule_via_ad(config, f, x)[2]
        v -> unthunk(pb(v)[2]), w -> pb(w)[1]
    end
    return fᴴ, construct∂f_lin
end
function lin_preprocess(config, A::AbstractMatrix, x)
    fᴴ = adjoint(A)
    if A isa StridedMatrix
        construct∂f_lin = w -> InplaceableThunk(
            Ā -> _buildĀ_lin!(Ā, x, w),
            @thunk(_buildĀ_lin!(zero(A), x, w))
        )
    else
        construct∂f_lin = let project_A = ProjectTo(A)
            w -> @thunk(project_A(_buildĀ_lin!(zero(A), x, w)))
        end
    end
    return fᴴ, construct∂f_lin
end
function _buildĀ_lin!(Ā, v, w)
    if !(w isa AbstractZero)
        if eltype(Ā) <: Real && eltype(w) <: Complex
            mul!(Ā, _realview(w), _realview(v)', +1, +1)
            mul!(Ā, _imagview(w), _imagview(v)', +1, +1)
        else
            mul!(Ā, w, v', +1, 1)
        end
    end
    return Ā
end

# frule - currently untested - commented out while untested and unused

# function ChainRulesCore.frule((_, ΔA, Δb, Δx₀, _, Δa₀, Δa₁)::Tuple, ::typeof(linsolve),
#                               A::AbstractMatrix, b::AbstractVector, x₀, algorithm, a₀, a₁)
#     (x, info) = linsolve(A, b, x₀, algorithm, a₀, a₁)

#     if Δb isa ChainRulesCore.AbstractZero
#         rhs = zerovector(b)
#     else
#         rhs = scale(Δb, (1 - Δa₁))
#     end
#     if !iszero(Δa₀)
#         rhs = add!!(rhs, x, -Δa₀)
#     end
#     if !iszero(ΔA)
#         rhs = mul!(rhs, ΔA, x, -a₁, true)
#     end
#     (Δx, forward_info) = linsolve(A, rhs, zerovector(rhs), algorithm, a₀, a₁)
#     if info.converged > 0 && forward_info.converged == 0 && alg_rrule.verbosity >= SILENT_LEVEL
#         @warn "The tangent linear problem did not converge, whereas the primal linear problem did."
#     end
#     return (x, info), (Δx, NoTangent())
# end

# function ChainRulesCore.frule(config::RuleConfig{>:HasForwardsMode}, tangents,
#                               ::typeof(linsolve),
#                               A::AbstractMatrix, b::AbstractVector, x₀, algorithm, a₀, a₁)
#     return frule(tangents, linsolve, A, b, x₀, algorithm, a₀, a₁)
# end

# function ChainRulesCore.frule(config::RuleConfig{>:HasForwardsMode},
#                               (_, Δf, Δb, Δx₀, _, Δa₀, Δa₁),
#                               ::typeof(linsolve),
#                               f, b, x₀, algorithm, a₀, a₁)
#     (x, info) = linsolve(f, b, x₀, algorithm, a₀, a₁)

#     if Δb isa AbstractZero
#         rhs = zerovector(b)
#     else
#         rhs = scale(Δb, (1 - Δa₁))
#     end
#     if !iszero(Δa₀)
#         rhs = add!!(rhs, x, -Δa₀)
#     end
#     if !(Δf isa AbstractZero)
#         rhs = add!!(rhs, frule_via_ad(config, (Δf, ZeroTangent()), f, x), -a₀)
#     end
#     (Δx, forward_info) = linsolve(f, rhs, zerovector(rhs), algorithm, a₀, a₁)
#     if info.converged > 0 && forward_info.converged == 0 && alg_rrule.verbosity >= SILENT_LEVEL
#         @warn "The tangent linear problem did not converge, whereas the primal linear problem did."
#     end
#     return (x, info), (Δx, NoTangent())
# end
