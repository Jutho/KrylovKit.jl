function geneigsolve(f, x₀, howmany::Int, which::Selector, alg::GolubYe)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    ax₀, bx₀ = genapply(f, x₀)
    numops = 1
    β₀ = norm(x₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    xax = inner(x₀, ax₀) / β₀^2
    xbx = inner(x₀, bx₀) / β₀^2
    T = promote_type(typeof(xax), typeof(xbx))
    invβ₀ = one(T) / β₀
    v = scale(x₀, invβ₀) # v = mul!(similar(x₀, T), x₀, invβ₀)
    av = scale!(zerovector(v), ax₀, invβ₀)
    bv = scale!(zerovector(v), bx₀, invβ₀)
    ρ = checkhermitian(xax) / checkposdef(xbx)
    r = add!(av, bv, -ρ)
    tol::typeof(ρ) = alg.tol

    # allocate storage
    HHA = fill(zero(T), krylovdim + 1, krylovdim + 1)
    HHB = fill(zero(T), krylovdim + 1, krylovdim + 1)

    # Start Lanczos iteration with A - ρ ⋅ B
    V = OrthonormalBasis([v])
    BV = [bv]
    sizehint!(V, krylovdim + 1)
    sizehint!(BV, krylovdim + 1)

    r, α = orthogonalize!(r, v, alg.orth) # α should be zero, otherwise ρ was miscalculated
    β = norm(r)
    m = 1
    HHA[m, m] = real(α)

    while m < krylovdim
        v = scale!(r, 1 / β)
        push!(V, r)
        HHA[m + 1, m] = β
        HHA[m, m + 1] = β
        βold = β
        r, α, β, bv = golubyerecurrence(f, ρ, V, βold, alg.orth)
        numops += 1
        m += 1
        n = hypot(α, β, βold)
        HHA[m, m] = checkhermitian(α, n)
        push!(BV, bv)

        if alg.verbosity > 2
            @info "Golub Ye iteration step $m: normres = $β"
        end
        β < tol && m >= howmany && break
    end

    # Process
    HA = view(HHA, 1:m, 1:m)
    HB = view(HHB, 1:m, 1:m)
    buildHB!(HB, V, BV)
    HA .+= ρ .* HB

    D, Z = geneigh!(HA, HB)
    by, rev = eigsort(which)
    p = sortperm(D; by=by, rev=rev)
    xold = V[1]

    z = view(Z, :, p[1])
    x = mul!(zerovector(xold), V, z)
    ax, bx = genapply(f, x)
    numops += 1
    ρ = checkhermitian(inner(x, ax)) / checkposdef(inner(x, bx))
    r = add!(ax, bx, -ρ)
    normr = norm(r)

    converged = 0
    values = Vector{typeof(ρ)}(undef, 0)
    vectors = Vector{typeof(x)}(undef, 0)
    residuals = Vector{typeof(r)}(undef, 0)
    normresiduals = Vector{typeof(normr)}(undef, 0)
    while normr <= tol && converged < howmany
        push!(values, ρ)
        push!(vectors, x)
        push!(residuals, r)
        push!(normresiduals, normr)
        converged += 1

        z = view(Z, :, p[converged + 1])
        x = mul!(zerovector(xold), V, z)
        ax, bx = genapply(f, x)
        numops += 1
        ρ = checkhermitian(inner(x, ax)) / checkposdef(inner(x, bx))
        r = add!(ax, bx, -ρ)
        normr = norm(r)
    end

    if alg.verbosity > 1
        msg = "Golub-Ye generalized eigsolve in iter $numiter: "
        msg *= "$converged values converged: $ρ, normres = ("
        for i in 1:converged
            msg *= @sprintf("%.2e", normresiduals[i])
            msg *= ", "
        end
        if converged < howmany
            msg *= @sprintf("%.2e", normr)
        end
        msg *= ")"
        @info msg
    end

    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        numiter += 1
        fill!(HHA, zero(T))
        fill!(HHB, zero(T))
        resize!(V, 0)
        resize!(BV, 0)

        # Start Lanczos iteration with A - ρ ⋅ B
        invβ = 1 / norm(x)
        v = scale!(x, invβ)
        bv = scale!(bx, invβ)
        r = scale!(r, invβ)
        r, α = orthogonalize!(r, v, alg.orth) # α should be zero, otherwise ρ was miscalculated
        β = norm(r)
        m = 1
        push!(V, v)
        HHA[m, m] = real(α)
        push!(BV, bv)

        while m < krylovdim - converged
            v = scale!(r, 1 / β)
            push!(V, r)
            HHA[m + 1, m] = β
            HHA[m, m + 1] = β
            βold = β
            r, α, β, bv = golubyerecurrence(f, ρ, V, βold, alg.orth)
            numops += 1
            m += 1
            n = hypot(α, β, βold)
            HHA[m, m] = checkhermitian(α, n)
            push!(BV, bv)

            if alg.verbosity > 2
                @info "Golub Ye iteration step $m: normres = $β"
            end
            β < tol && m >= howmany && break
        end

        # add xold
        v, = orthonormalize!(xold, V, alg.orth)
        av, bv = genapply(f, v)
        numops += 1
        av = add!(av, bv, -ρ)
        for i in 1:m
            HHA[i, m + 1] = inner(V[i], av)
            HHA[m + 1, i] = conj(HHA[i, m + 1])
        end
        m += 1
        HHA[m, m] = checkhermitian(inner(v, av))
        push!(V, v)
        push!(BV, bv)

        # add converged vectors
        @inbounds for i in 1:converged
            v, = orthonormalize(vectors[i], V, alg.orth)
            av, bv = genapply(f, v)
            numops += 1
            av = add!(av, bv, -ρ)
            for i in 1:m
                HHA[i, m + 1] = inner(V[i], av)
                HHA[m + 1, i] = conj(HHA[i, m + 1])
            end
            m += 1
            HHA[m, m] = checkhermitian(inner(v, av))
            push!(V, v)
            push!(BV, bv)
        end

        # Process
        HA = view(HHA, 1:m, 1:m)
        HB = view(HHB, 1:m, 1:m)
        buildHB!(HB, V, BV)
        HA .+= ρ .* HB

        D, Z = geneigh!(HA, HB)
        by, rev = eigsort(which)
        p = sortperm(D; by=by, rev=rev)
        xold = V[1]

        converged = 0
        resize!(values, 0)
        resize!(vectors, 0)
        resize!(residuals, 0)
        resize!(normresiduals, 0)
        while true
            z = view(Z, :, p[converged + 1])
            x = mul!(zerovector(xold), V, z)
            ax, bx = genapply(f, x)
            numops += 1
            ρ = checkhermitian(inner(x, ax)) / checkposdef(inner(x, bx))
            r = add!(ax, bx, -ρ)
            normr = norm(r)

            if normr > tol || converged >= howmany
                break
            end

            push!(values, ρ)
            push!(vectors, x)
            push!(residuals, r)
            push!(normresiduals, normr)
            converged += 1
        end

        if alg.verbosity > 1
            msg = "Golub-Ye generalized eigsolve in iter $numiter: "
            msg *= "$converged values converged: $ρ, normres = ("
            for i in 1:converged
                msg *= @sprintf("%.2e", normresiduals[i])
                msg *= ", "
            end
            if converged < howmany
                msg *= @sprintf("%.2e", normr)
            end
            msg *= ")"
            @info msg
        end
    end

    if converged > howmany
        howmany = converged
    end
    for k in (converged + 1):howmany
        z = view(Z, :, p[k])
        x = mul!(zerovector(xold), V, z)
        ax, bx = genapply(f, x)
        numops += 1
        ρ = checkhermitian(inner(x, ax)) / checkposdef(inner(x, bx))
        r = add!(ax, bx, -ρ)
        normr = norm(r)

        push!(values, ρ)
        push!(vectors, x)
        push!(residuals, r)
        push!(normresiduals, normr)
    end

    if alg.verbosity > 0
        if converged < howmany
            @warn """GolubYe eigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Lanczos eigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end

    return values,
           vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!(av, bv, -ρ)
    α = inner(v, w)

    w = add!(w, V[end - 1], -β)
    w = add!(w, v, -α)
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!(av, bv, -ρ)
    w = add!(w, V[end - 1], -β)

    w, α = orthogonalize!(w, v, orth)
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt2)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!(av, bv, -ρ)
    α = inner(v, w)
    w = add!(w, V[end - 1], -β)
    w = add!(w, v, -α)

    w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
    α += s[end]
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt2)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!(av, bv, -ρ)
    w = add!(w, V[end - 1], -β)
    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())

    s = α
    for q in V
        w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
    end
    α += s
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidtIR)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!(av, bv, -ρ)
    α = inner(v, w)
    w = add!(w, V[end - 1], -β)
    w = add!(w, v, -α)

    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
        α += s[end]
        β = norm(w)
    end
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidtIR)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!(av, bv, -ρ)
    w = add!(w, V[end - 1], -β)

    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        s = zero(α)
        for q in V
            w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
        end
        α += s
        β = norm(w)
    end
    return w, α, β, bv
end

function buildHB!(HB, V, BV)
    m = length(V)
    @inbounds for j in 1:m
        HB[j, j] = checkposdef(inner(V[j], BV[j]))
        for i in (j + 1):m
            HB[i, j] = inner(V[i], BV[j])
            HB[j, i] = conj(HB[i, j])
        end
    end
end
