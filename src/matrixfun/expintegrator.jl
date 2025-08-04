"""
    function expintegrator(A, t::Number, u₀, u₁, …; kwargs...)
    function expintegrator(A, t::Number, (u₀, u₁, …); kwargs...)
    function expintegrator(A, t::Number, (u₀, u₁, …), algorithm)

Compute ``y = ϕ₀(t*A)*u₀ + t*ϕ₁(t*A)*u₁ + t^2*ϕ₂(t*A)*u₂ + …``, where `A` is a general
linear map, i.e. a `AbstractMatrix` or just a general function or callable object and `u₀`,
`u₁` are of any Julia type with vector like behavior. Here, ``ϕ₀(z) = exp(z)`` and
``ϕⱼ₊₁ = (ϕⱼ(z) - 1/j!)/z``. In particular, ``y = x(t)`` represents the solution of the ODE
``ẋ(t) = A*x(t) + ∑ⱼ t^j/j! uⱼ₊₁`` with ``x(0) = u₀``.

!!! note

    When there are only input vectors `u₀` and `u₁`, `t` can equal `Inf`, in which the
    algorithm tries to evolve all the way to the fixed point `y = - A \\ u₁ + P₀ u₀` with
    `P₀` the projector onto the eigenspace of eigenvalue zero (if any) of `A`. If `A` has
    any eigenvalues with real part larger than zero, however, the solution to the ODE will
    diverge, i.e. the fixed point is not stable.

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

  - `y`: the result of the computation, i.e.
    ``y = ϕ₀(t*A)*u₀ + t*ϕ₁(t*A)*u₁ + t^2*ϕ₂(t*A)*u₂ + …``

  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: 0 or 1 if the solution `y` at time `t` was found with an
        error below the requested tolerance per unit time, i.e. if `info.normres <= tol * abs(t)`
      + `info.residual::Nothing`: value `nothing`, there is no concept of a residual in
        this case
      + `info.normres::Real`: a (rough) estimate of the total error accumulated in the
        solution
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
      + `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

### Keyword arguments:

Keyword arguments and their default values are given by:

  - `verbosity::Int = SILENT_LEVEL`: verbosity level, i.e. 
    - SILENT_LEVEL (suppress all messages)
    - WARN_LEVEL (only warnings)
    - STARTSTOP_LEVEL (one message with convergence info at the end)
    - EACHITERATION_LEVEL (progress info after every iteration)
    - EACHITERATION_LEVEL+ (all of the above and additional information about the Lanczos or Arnoldi iteration)
  - `krylovdim = 30`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should
    not necessarily support the `Base.length` function. If you know the actual problem
    dimension is smaller than the default value, it is useful to reduce the value of
    `krylovdim`, though in principle this should be detected.
  - `tol = 1e-12`: the requested accuracy per unit time, i.e. if you want a certain
    precision `ϵ` on the final result, set `tol = ϵ/abs(t)`. If you work in e.g. single
    precision (`Float32`), you should definitely change the default value.
  - `maxiter::Int = 100`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
  - `issymmetric`: if the linear map is symmetric, only meaningful if `T<:Real`
  - `ishermitian`: if the linear map is hermitian
    The default value for the last two depends on the method. If an `AbstractMatrix` is
    used, `issymmetric` and `ishermitian` are checked for that matrix, otherwise the default
    values are `issymmetric = false` and `ishermitian = T <: Real && issymmetric`.
  - `eager::Bool = false`: if true, eagerly try to compute the result after every expansion
    of the Krylov subspace to test for convergence, otherwise wait until the Krylov subspace
    as dimension `krylovdim`. This can result in a faster return, for example if the total
    time for the evolution is quite small, but also has some overhead, as more computations
    are performed after every expansion step.

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
    Ts = typeof.(inner.((u₀, us...), (u₀,)))
    T = promote_type(typeof(t), Ts...)
    alg = eigselector(A, T; kwargs...)
    return expintegrator(A, t, (u₀, us...), alg)
end

function expintegrator(A, t::Number, u::Tuple, alg::Union{Lanczos, Arnoldi})
    length(u) == 1 && return expintegrator(A, t, (u[1], zerovector(u[1])), alg)

    p = length(u) - 1

    # process initial vector and determine result type
    u₀ = first(u)
    β₀ = norm(u₀)
    Au₀ = apply(A, u₀) # used to determine return type
    numops = 1
    T = promote_type(typeof(t), (typeof.(inner.(u, (Au₀,))))...)
    S = real(T)
    w₀ = scale(u₀, one(T))

    # maxiter, krylovdim and related allocations
    maxiter = alg.maxiter
    @assert maxiter >= 1
    krylovdim = alg.krylovdim
    K = krylovdim
    HH = zeros(T, (krylovdim + p + 1, krylovdim + p + 1))

    # time step parameters
    η::S = alg.tol # tol is per unit time
    totalerr = zero(η)
    sgn = sign(t)
    τ::S = abs(t)
    if isfinite(τ)
        Δτ = τ  # don't try any clever initial guesses, rely on correction mechanism
        Δτmin = τ / alg.maxiter
        maxerr = τ * η
    else
        Δτ = oneunit(S)
        Δτmin = zero(S)
        maxerr = η
    end
    totaltimestring = @sprintf("%.2e", τ)

    # safety factors
    δ::S = 1.2
    γ::S = 0.8

    # initial vectors
    τ₀ = zero(τ)
    w = Vector{typeof(w₀)}(undef, p + 1)
    w[1] = w₀
    # reuse the result of apply computed earlier:
    w[2] = scale!!(zerovector(w₀), Au₀, one(T))
    for j in 1:p
        if j > 1
            w[j + 1] = apply(A, w[j])
            numops += 1
        end
        lfac = 1
        for l in 0:(p - j)
            w[j + 1] = add!!(w[j + 1], u[j + l + 1], (sgn * τ₀)^l / lfac)
            lfac *= l + 1
        end
    end
    v = zerovector(w₀)
    β = norm(w[p + 1])
    if β < η && p == 1
        if alg.verbosity >= STARTSTOP_LEVEL
            @info "expintegrate finished after 0 iterations, converged to fixed point up to error = $(normres2string(β))"
        end
        return w₀, ConvergenceInfo(1, nothing, β, 0, numops)
    end
    v = scale!!(v, w[p + 1], 1 / β)

    # initialize iterator
    if alg isa Lanczos
        iter = LanczosIterator(A, w[p + 1], alg.orth)
    else
        iter = ArnoldiIterator(A, w[p + 1], alg.orth)
    end
    fact = initialize(iter; verbosity = alg.verbosity)
    numops += 1
    sizehint!(fact, krylovdim)

    # start outer iteration loop
    numiter = 1
    while true
        K = length(fact)
        V = basis(fact)

        if K == krylovdim
            if numiter < maxiter
                Δτ = min(Δτ, τ - τ₀)
                if isfinite(τ) # try to adapt minimal time step
                    Δτmin = (τ - τ₀) / (maxiter - numiter + 1)
                end
            else
                Δτ = τ - τ₀
            end

            # Small matrix exponential and error estimation
            H = fill!(view(HH, 1:(K + p + 1), 1:(K + p + 1)), zero(T))
            mul!(view(H, 1:K, 1:K), rayleighquotient(fact), sgn * Δτ)
            H[1, K + 1] = 1
            for i in 1:p
                H[K + i, K + i + 1] = 1
            end
            expH = exp(H) # LinearAlgebra.exp! is type unstable for SubArray instances
            ϵ = abs(Δτ^p * β * normres(fact) * expH[K, K + p + 1])
            ω = ϵ / (Δτ * η)

            q::S = K / 2
            while numiter < maxiter && ω >= one(ω) && Δτ > Δτmin
                ϵ_prev = ϵ
                Δτ_prev = Δτ
                Δτ = max(Δτ * (γ / ω)^(1 / (q + 1)), Δτmin)
                H = fill!(view(HH, 1:(K + p + 1), 1:(K + p + 1)), zero(T))
                mul!(view(H, 1:K, 1:K), rayleighquotient(fact), sgn * Δτ)
                H[1, K + 1] = 1
                for i in 1:p
                    H[K + i, K + i + 1] = 1
                end
                expH = exp(H) # LinearAlgebra.exp! is type unstable for SubArray instances
                ϵ = abs(Δτ^p * β * normres(fact) * expH[K, K + p + 1])
                ω = ϵ / (Δτ * η)
                q = max(zero(q), log(ϵ / ϵ_prev) / log(Δτ / Δτ_prev) - 1)
            end

            # take time step
            τ₀ = numiter < maxiter ? τ₀ + Δτ : τ # to avoid floating point errors
            totalerr += ϵ
            jfac = 1
            for j in 1:(p - 1)
                w₀ = add!!(w₀, w[j + 1], (sgn * Δτ)^j / jfac)
                jfac *= (j + 1)
            end
            w[p + 1] = unproject!!(w[p + 1], basis(fact), view(expH, 1:K, K + p))
            # add first correction
            w[p + 1] = add!!(w[p + 1], residual(fact), expH[K, K + p + 1])
            w₀ = add!!(w₀, w[p + 1], β * (sgn * Δτ)^p)
            w[1] = w₀

            # increase time step for next iteration:
            if ω < γ
                Δτ *= (γ / ω)^(1 / (q + 1))
            end
        elseif normres(fact) <= ((τ - τ₀) * η) || alg.eager
            # Small matrix exponential and error estimation
            H = fill!(view(HH, 1:(K + p + 1), 1:(K + p + 1)), zero(T))
            mul!(view(H, 1:K, 1:K), rayleighquotient(fact), sgn * (τ - τ₀))
            H[1, K + 1] = 1
            for i in 1:p
                H[K + i, K + i + 1] = 1
            end
            expH = exp(H) # LinearAlgebra.exp! is type unstable for SubArray instances
            ϵ = abs((τ - τ₀)^p * β * normres(fact) * expH[K, K + p + 1])
            ω = ϵ / ((τ - τ₀) * η)
            if ω < one(ω)
                # take time step
                totalerr += ϵ
                jfac = 1
                for j in 1:(p - 1)
                    w₀ = add!!(w₀, w[j + 1], (sgn * (τ - τ₀))^j / jfac)
                    jfac *= (j + 1)
                end
                w[p + 1] = unproject!!(w[p + 1], basis(fact), view(expH, 1:K, K + p))
                # add first correction
                w[p + 1] = add!!(w[p + 1], residual(fact), expH[K, K + p + 1])
                w₀ = add!!(w₀, w[p + 1], β * (sgn * (τ - τ₀))^p)
                w[1] = w₀
                τ₀ = τ
            end
        end
        if τ₀ >= τ
            if totalerr <= maxerr
                if alg.verbosity >= STARTSTOP_LEVEL
                    @info """expintegrate finished after $numiter iterations:
                    * total error = $(normres2string(totalerr))
                    * number of operations = $numops"""
                end
                return w₀, ConvergenceInfo(1, nothing, totalerr, numiter, numops)
            else
                if alg.verbosity >= WARN_LEVEL
                    @warn """expintegrate did not reach sufficiently small error after $numiter iterations:
                    * total error = $(normres2string(totalerr))
                    * number of operations = $numops"""
                end
                return w₀, ConvergenceInfo(0, nothing, totalerr, numiter, numops)
            end
        end
        if K < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity)
            numops += 1
        else
            for j in 1:p
                w[j + 1] = apply(A, w[j])
                numops += 1
                lfac = 1
                for l in 0:(p - j)
                    w[j + 1] = add!!(w[j + 1], u[j + l + 1], (sgn * τ₀)^l / lfac)
                    lfac *= l + 1
                end
            end
            β = norm(w[p + 1])
            if β < η && p == 1 # w₀ is fixed point of ODE
                if alg.verbosity >= STARTSTOP_LEVEL
                    @info "expintegrate finished after $numiter iterations, converged to fixed point up to error = $(normres2string(totalerr))"
                end
                return w₀, ConvergenceInfo(1, nothing, β, numiter, numops)
            end
            v = scale!!(v, w[p + 1], 1 / β)

            if alg.verbosity >= EACHITERATION_LEVEL
                timestring = @sprintf("%.2e", τ₀)
                @info "expintegrate in iteration $numiter: reached time $timestring of $totaltimestring, total error = $(normres2string(totalerr))"
            end

            if alg isa Lanczos
                iter = LanczosIterator(A, w[p + 1], alg.orth)
            else
                iter = ArnoldiIterator(A, w[p + 1], alg.orth)
            end
            fact = initialize!(iter, fact; verbosity = alg.verbosity)
            numops += 1
            numiter += 1
        end
    end
    return
end
