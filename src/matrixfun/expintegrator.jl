"""
    function expintegrator(A, t::Number, u₀, u₁, …; kwargs...)
    function expintegrator(A, t::Number, (u₀, u₁, …); kwargs...)
    function expintegrator(A, t::Number, (u₀, u₁, …), algorithm)

Compute ``y = ϕ₀(t*A)*u₀ + t*ϕ₁(t*A)*u₁ + t^2*ϕ₂(t*A)*u₂ + …``, where `A` is a general linear map, i.e. a `AbstractMatrix` or just a general function or callable object and `u₀`, `u₁` are of any Julia type with vector like behavior. Here, ``ϕ₀(z) = exp(z)`` and ``ϕⱼ₊₁ = (ϕⱼ(z) - 1/j!)/z``. In particular, ``y = x(t)`` represents the solution of the ODE
``x' = A*x + ∑ⱼ t^j/j! uⱼ₊₁`` with ``x(0) = u₀``.

!!! note
    When there are only input vectors `u₀` and `u₁`, `t` can equal `Inf`, in which the
    algorithm tries to evolve all the way to the fixed point `y = - A \\ u₁ + P₀ u₀` with
    `P₀` the projector onto the eigenspace of eigenvalue zero (if any) of `A`. If `A` has
    any eigenvalues with real part larger than zero, however, the solution to the ODE will
    diverge, i.e. the fixed point is not stable.

!!! warning
    The returned solution might be the solution of the ODE integrated up to a smaller time ``t̃ = sign(t) * |t̃|`` with ``|t̃| < |t|``, when the required precision could not be attained. Always check `info.converged > 0` or `info.residual == 0` (see below).

### Arguments:
The linear map `A` can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object that implements the action of the linear map on a vector. If `A` is an
`AbstractMatrix`, `x` is expected to be an `AbstractVector`, otherwise `x` can be of any
type that behaves as a vector and supports the required methods (see KrylovKit docs).

The time parameter `t` can be real or complex, and it is better to choose `t` e.g. imaginary
and `A` hermitian, then to absorb the imaginary unit in an antihermitian `A`. For the
former, the Lanczos scheme is used to built a Krylov subspace, in which an approximation to
the exponential action of the linear map is obtained. The arguments `u₀`, `u₁`, … can be
of any type and should be in the domain of `A`.

### Return values:
The return value is always of the form `y, info = expintegrator(...)` with
*   `y`: the result of the computation, i.e.
    ``y = ϕ₀(t̃*A)*u₀ + t̃*ϕ₁(t̃*A)*u₁ + t̃^2*ϕ₂(t̃*A)*u₂ + …``
    with ``t̃ = sign(t) * |t̃|`` with ``|t̃| <= |t|``, such that the accumulated error in
    `y` per unit time is at most equal to the keyword argument `tol`
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: 0 or 1 if the solution `y` was evolved all the way up to the
        requested time `t`.
    -   `info.residual`: there is no residual in the conventional sense, however, this
        value equals the residual time `t - t̃`, i.e. it is zero if `info.converged == 1`
    -   `info.normres::Real`: a (rough) estimate of the total error accumulated in the
        solution, should be smaller than `tol * |t̃|`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

### Keyword arguments:
Keyword arguments and their default values are given by:
*   `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
*   `krylovdim = 30`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should
    not necessarily support the `Base.length` function. If you know the actual problem
    dimension is smaller than the default value, it is useful to reduce the value of
    `krylovdim`, though in principle this should be detected.
*   `tol = 1e-12`: the requested accuracy per unit time, i.e. if you want a certain
    precision `ϵ` on the final result, set `tol = ϵ/abs(t)`. If you work in e.g. single
    precision (`Float32`), you should definitely change the default value.
*   `maxiter::Int = 100`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
*   `issymmetric`: if the linear map is symmetric, only meaningful if `T<:Real`
*   `ishermitian`: if the linear map is hermitian
The default value for the last two depends on the method. If an `AbstractMatrix` is used,
`issymmetric` and `ishermitian` are checked for that matrix, ortherwise the default values
are `issymmetric = false` and `ishermitian = T <: Real && issymmetric`.

### Algorithm
The last method, without keyword arguments and the different vectors `u₀`, `u₁`, … in a
tuple, is the one that is finally called, and can also be used directly. Here, one
specifies the algorithm explicitly as either [`Lanczos`](@ref), for real symmetric or
complex hermitian linear maps, or [`Arnoldi`](@ref), for general linear maps. Note that
these names refer to the process for building the Krylov subspace, and that one can still
use complex time steps in combination with e.g. a real symmetric map.
"""
function expintegrator end

function expintegrator(A, t::Number, u₀, us...; kwargs...)
    alg = eigselector(A, promote_type(typeof(t), eltype(u₀), eltype.(us)...); kwargs...)
    expintegrator(A, t, (u₀, us...), alg)
end

function expintegrator(A, t::Number, u::Tuple, alg::Union{Lanczos,Arnoldi})
    length(u) == 1 && return expintegrator(A, t, (u[1], rmul!(similar(u[1]), false)), alg)

    p = length(u) - 1

    # process initial vector and determine result type
    u₀ = first(u)
    β₀ = norm(u₀)
    Au₀ = apply(A, u₀) # used to determine return type
    numops = 1
    T = promote_type(promote_type(eltype(Au₀), typeof(β₀), typeof(t)),
                        promote_type(eltype.(u)...))
    S = real(T)
    w₀ = copyto!(similar(u₀, T), u₀)

    # krylovdim and related allocations
    krylovdim = alg.krylovdim
    K = krylovdim
    HH = zeros(T, (krylovdim+p+1, krylovdim+p+1))

    # time step parameters
    η::S = alg.tol # tol is per unit time
    totalerr = zero(η)
    sgn = sign(t)
    τ::S = abs(t)
    Δτ::S = one(τ) # don't try any clever initial guesses, rely on correction mechanism
    τ₀ = zero(τ)

    # safety factors
    δ::S = 1.2
    γ::S = 0.8

    # initial vectors
    w = Vector{typeof(w₀)}(undef, p+1)
    w[1] = w₀
    for j = 1:p
        w[j+1] = apply(A, w[j])
        numops += 1
        lfac = 1
        for l = 0:p-j
            w[j+1] = axpy!((sgn*τ₀)^l/lfac, u[j+l+1], w[j+1])
            lfac *= l+1
        end
    end
    v = similar(w₀)
    β = norm(w[p+1])
    if β < alg.tol && p == 1
        if alg.verbosity > 0
            @info """expintegrate finished after 0 iterations, converged to fixed point up to error = $β"""
        end
        return w₀, ConvergenceInfo(1, zero(τ), β, 0, numops)
    end
    mul!(v, w[p+1], 1/β)

    # initialize iterator
    if alg isa Lanczos
        iter = LanczosIterator(A, w[p+1], alg.orth)
    else
        iter = ArnoldiIterator(A, w[p+1], alg.orth)
    end
    fact = initialize(iter; verbosity = alg.verbosity - 2)
    numops += 1
    sizehint!(fact, krylovdim)

    # start outer iteration loop
    maxiter = alg.maxiter
    numiter = 0
    while true
        if β < alg.tol && p == 1 # w₀ is fixed point of ODE
            if alg.verbosity > 0
                @info """expintegrate finished after $numiter iterations, converged to fixed point up to error = $β"""
            end
            return w₀, ConvergenceInfo(1, zero(τ), β, numiter, numops)
        end

        numiter += 1
        Δτ = min(Δτ, τ-τ₀)

        # Lanczos or Arnoldi factorization
        while normres(fact) > eps() && length(fact) < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
            numops += 1
        end
        K = fact.k # current Krylov dimension
        V = basis(fact)

        # Small matrix exponential and error estimation
        H = fill!(view(HH, 1:K+p+1, 1:K+p+1), zero(T))
        mul!(view(H, 1:K, 1:K), rayleighquotient(fact), sgn*Δτ)
        H[1, K+1] = 1
        for i = 1:p
            H[K+i, K+i+1] = 1
        end
        expH = LinearAlgebra.exp!(H)
        ϵ = abs(Δτ^p * β * normres(fact) * expH[K,K+p+1])
        ω = ϵ / (Δτ * η)

        q = K/2
        while ω > one(ω)
            ϵ_prev = ϵ
            Δτ_prev = Δτ
            Δτ *= (γ/ω)^(1/(q+1))
            H = fill!(view(HH, 1:K+p+1, 1:K+p+1), zero(T))
            mul!(view(H, 1:K, 1:K), rayleighquotient(fact), sgn*Δτ)
            H[1, K+1] = 1
            for i = 1:p
                H[K+i, K+i+1] = 1
            end
            expH = LinearAlgebra.exp!(H)
            ϵ = abs(Δτ^p * β * normres(fact) * expH[K,K+p+1])
            ω = ϵ / (Δτ * η)
            q = max(zero(q),  log(ϵ / ϵ_prev)/log(Δτ / Δτ_prev)-1)
        end

        # take time step
        totalerr += ϵ
        τ₀ += Δτ
        jfac = 1
        for j = 1:p-1
            w₀ = axpy!((sgn*Δτ)^j/jfac, w[j+1], w₀)
            jfac *= (j+1)
        end
        w[p+1] = mul!(w[p+1], basis(fact), view(expH, 1:K, K+p))
        # add first correction
        w[p+1] = axpy!(expH[K,K+p+1], residual(fact), w[p+1])
        w₀ = axpy!(β*(sgn*Δτ)^p, w[p+1], w₀)

        # increase time step for next iteration:
        if ω < γ
            Δτ *= (γ/ω)^(1/(q+1))
        end

        if alg.verbosity > 1
            msg = "expintegrate in iteration $numiter: "
            msg *= "reached time " * @sprintf("%.2e", τ₀)
            msg *= ", total error = " * @sprintf("%.4e", totalerr)
            @info msg
        end

        if τ₀ >= τ
            if alg.verbosity > 0
                @info """expintegrate finished after $numiter iterations: total error = $totalerr"""
            end
            return w₀, ConvergenceInfo(1, zero(τ), totalerr, numiter, numops)
        elseif numiter == maxiter
            if alg.verbosity > 0
                @warn """expintegrate finished without convergence after $numiter iterations:
                total error = $totalerr, residual time = $(τ - τ₀)"""
            end
            return w₀, ConvergenceInfo(0, τ-τ₀, totalerr, numiter, numops)
        else
            for j = 1:p
                w[j+1] = apply(A, w[j])
                numops += 1
                lfac = 1
                for l = 0:p-j
                    w[j+1] = axpy!((sgn*τ₀)^l/lfac, u[j+l+1], w[j+1])
                    lfac *= l+1
                end
            end
            β = norm(w[p+1])
            mul!(v, w[p+1], 1/β)
            if alg isa Lanczos
                iter = LanczosIterator(A, w[p+1], alg.orth)
            else
                iter = ArnoldiIterator(A, w[p+1], alg.orth)
            end
            fact = initialize!(iter, fact; verbosity = alg.verbosity-2)
            numops += 1
        end
    end
end
