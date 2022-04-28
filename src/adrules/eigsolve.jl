function ChainRulesCore.rrule(
    ::typeof(eigsolve),
    A::AbstractMatrix,
    x₀,
    howmany,
    which,
    alg::Lanczos
)
    vals, vecs, info = eigsolve(A, x₀, howmany, which, alg)

    # manually truncate "howmany" values
    vals = vals[1:howmany]
    vecs = vecs[1:howmany]

    project_A = ProjectTo(A)

    function eigsolve_pullback(ΔX)
        Δvals, Δvecs, _ = ΔX
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂alg = NoTangent()
        ∂which = NoTangent()
        ∂howmany = NoTangent()
        ∂A = mapreduce(+, vals, vecs, Δvals, Δvecs) do λ, v, Δλ, Δv
            if isa(Δv, typeof(ZeroTangent()))
                ξ = Δv
            else
                alg_reverse = GMRES(;
                    tol=alg.tol,
                    krylovdim=alg.krylovdim,
                    maxiter=alg.maxiter,
                    orth=alg.orth
                )
                b = Δv - dot(v, Δv) * v
                ξ, info_reverse = linsolve(A, b, zero(λ) * b, alg_reverse, -λ)
                info_reverse.converged == 0 && @warn "Cotangent problem did not converge."
                ξ -= dot(v, ξ) * v
            end
            if A isa StridedMatrix
                return InplaceableThunk(
                    Ā -> mul!(Ā, v, Δλ * v' - ξ', true, true),
                    @thunk(v * (Δλ * v' - ξ')),
                )
            else
                return @thunk(project_A(v * (Δλ * v' - ξ')))
            end
        end
        return ∂self, ∂A, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, vecs, info), eigsolve_pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(eigsolve),
    f,
    x₀,
    howmany,
    which,
    alg::Lanczos
)
    vals, vecs, info = eigsolve(f, x₀, howmany, which, alg)

    # truncate "howmany" values
    vals = vals[1:howmany]
    vecs = vecs[1:howmany]

    f_pullbacks = map(x -> rrule_via_ad(config, f, x)[2], vecs)

    function eigsolve_pullback(ΔX)
        Δvals, Δvecs, _ = unthunk(ΔX)
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂alg = NoTangent()
        ∂which = NoTangent()
        ∂howmany = NoTangent()
        ∂f = mapreduce(+, vals, vecs, Δvals, Δvecs, f_pullbacks) do λ, v, Δλ, Δv, f_pullback
            if isa(Δv, typeof(ZeroTangent()))
                ξ = Δv
            else
                alg_reverse = GMRES(;
                    tol=alg.tol,
                    krylovdim=alg.krylovdim,
                    maxiter=alg.maxiter,
                    orth=alg.orth
                )
                b = axpy!(-dot(v, Δv), v, one(λ) * Δv)
                ξ, info_reverse = linsolve(f, b, zero(λ) * v, alg_reverse, -λ)
                info_reverse.converged == 0 && @warn "Cotangent problem did not converge."
                axpy!(-dot(v, ξ), v, ξ)
            end
            axpby!(Δλ, v, -one(λ), ξ)
            return f_pullback(ξ)[1]
        end
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, vecs, info), eigsolve_pullback
end

# correct dispatch behaviour
function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(eigsolve),
    A::AbstractMatrix,
    x₀,
    howmany,
    which,
    alg::Lanczos
)
    return rrule(eigsolve, A, x₀, howmany, which, alg)
end

function ChainRulesCore.rrule(
    ::typeof(eigsolve),
    A::AbstractMatrix,
    x₀,
    howmany,
    which,
    alg::Arnoldi
)
    λᵣs, rs, infoᵣ = eigsolve(A, x₀, howmany, which, alg)
    λᵣs = λᵣs[1:howmany]
    rs = rs[1:howmany]

    # compute left eigenvectors
    λₗs, ls, infoₗ = eigsolve(A', x₀, howmany, which, alg)
    infoₗ.converged < howmany && @warn "Left eigenvectors not converged."
    λₗs = λₗs[1:howmany]
    by, rev = eigsort(which)
    p = sortperm(conj.(λₗs) .+ alg.tol * 1.0im, by=by, rev=rev)
    all(conj.(λₗs[p]) .≈ λᵣs) || @warn "Left and right eigenvalues disagree."
    ls = ls[p]

    project_A = ProjectTo(A)

    function eigsolve_pullback(ΔX)
        Δλs, Δrs, _ = ΔX
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂alg = NoTangent()
        ∂which = NoTangent()
        ∂howmany = NoTangent()
        ∂A = mapreduce(+, λᵣs, rs, ls, Δλs, Δrs) do λ, r, l, Δλ, Δr
            ϕ = dot(l, r)
            Δr -= dot(r, Δr) * r
            if isa(Δr, typeof(ZeroTangent()))
                ξ = Δr
            else
                alg_reverse = GMRES(; tol=alg.tol, krylovdim=alg.krylovdim, maxiter=alg.maxiter, orth=alg.orth)
                b = Δr - dot(r, Δr) / conj(ϕ) * l
                ξ, info_reverse = linsolve(A', b, zero(λ) * b, alg_reverse, -conj(λ))
                info_reverse.converged == 0 && @warn "Cotangent problem did not converge."
                ξ -= dot(r, ξ) / conj(ϕ) * l
            end
            if A isa StridedMatrix
                return InplaceableThunk(
                    Ā -> mul!(Ā, Δλ / conj(ϕ) * l - ξ, r', true, true),
                    @thunk((Δλ / conj(ϕ) * l - ξ) * r'),
                )
            else
                return project_A((Δλ / conj(ϕ) * l - ξ) * r')
            end
        end
        return ∂self, ∂A, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (λᵣs, rs, infoᵣ), eigsolve_pullback
end

# correct dispatch behaviour
function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(eigsolve),
    A::AbstractMatrix,
    x₀,
    howmany,
    which,
    alg::Arnoldi
)
    return rrule(eigsolve, A, x₀, howmany, which, alg)
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(eigsolve),
    f,
    x₀,
    howmany,
    which,
    alg::Arnoldi
)
    λᵣs, rs, infoᵣ = eigsolve(f, x₀, howmany, which, alg)
    λᵣs = λᵣs[1:howmany]
    rs = rs[1:howmany]

    # compute left eigenvectors
    (_, f_pullback) = rrule_via_ad(config, f, x₀)
    fᴴ(x) = f_pullback(x)[2]
    λₗs, ls, infoₗ = eigsolve(fᴴ, x₀, howmany, which, alg)
    infoₗ.converged < howmany && @warn "Left eigenvectors not converged."
    λₗs = λₗs[1:howmany]
    by, rev = eigsort(which)
    p = sortperm(conj.(λₗs) .+ alg.tol * 1.0im, by=by, rev=rev)
    all(conj.(λₗs[p]) .≈ λᵣs) || @warn "Left and right eigenvalues disagree."
    ls = ls[p]

    f_pullbacks = map(x -> rrule_via_ad(config, f, x)[2], rs)

    function eigsolve_pullback(ΔX)
        Δλs, Δrs, _ = ΔX
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂alg = NoTangent()
        ∂which = NoTangent()
        ∂howmany = NoTangent()
        ∂f = mapreduce(+, λᵣs, rs, ls, Δλs, Δrs, f_pullbacks) do λ, r, l, Δλ, Δr, back
            ϕ = dot(l, r)
            if isa(Δr, typeof(ZeroTangent()))
                ξ = Δr
            else
                axpy!(-dot(r, Δr), r, Δr)
                alg_reverse = GMRES(;
                    tol=alg.tol,
                    krylovdim=alg.krylovdim,
                    maxiter=alg.maxiter,
                    orth=alg.orth
                )
                b = axpy!(-dot(r, Δr) / conj(ϕ), l, Δr)
                ξ, info_reverse = linsolve(fᴴ, b, zero(λ) * r, alg_reverse, -conj(λ))
                info_reverse.converged == 0 && @warn "Cotangent problem did not converge."
                axpy!(-dot(r, ξ) / conj(ϕ), l, ξ)
            end
            return back(axpby!(Δλ / conj(ϕ), l, -one(Δλ), ξ))[1]
        end
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (λᵣs, rs, infoᵣ), eigsolve_pullback
end
