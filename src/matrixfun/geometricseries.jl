# """
#     function exponentiate(A, t::Number, x; kwargs...)
#     function exponentiate(A, t::Number, x, algorithm)
#
# Compute ``y = exp(t*A) x``, where `A` is a general linear map, i.e. a `AbstractMatrix` or
# just a general function or callable object and `x` is of any Julia type with vector like
# behavior.
#
# ### Arguments:
# The linear map `A` can be an `AbstractMatrix` (dense or sparse) or a general function or
# callable object that implements the action of the linear map on a vector. If `A` is an
# `AbstractMatrix`, `x` is expected to be an `AbstractVector`, otherwise `x` can be of any
# type that behaves as a vector and supports the required methods (see KrylovKit docs).
#
# The time parameter `t` can be real or complex, and it is better to choose `t` e.g. imaginary
# and `A` hermitian, then to absorb the imaginary unit in an antihermitian `A`. For the
# former, the Lanczos scheme is used to built a Krylov subspace, in which an approximation to
# the exponential action of the linear map is obtained. The argument `x` can be of any type
# and should be in the domain of `A`.
#
#
# ### Return values:
# The return value is always of the form `y, info = eigsolve(...)` with
# *   `y`: the result of the computation, i.e. `y = exp(t*A)*x`
# *   `info`: an object of type [`ConvergenceInfo`], which has the following fields
#     -   `info.converged::Int`: 0 or 1 if the solution `y` was approximated up to the
#         requested tolerance `tol`.
#     -   `info.residual::Nothing`: value `nothing`, there is no concept of a residual in
#         this case
#     -   `info.normres::Real`: an estimate (upper bound) of the error between the
#         approximate and exact solution
#     -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
#         `f` was called, or a vector was multiplied with `A`
#     -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
# !!! warning "Check for convergence"
#     No warning is printed if not all requested eigenvalues were converged, so always check
#     if `info.converged >= howmany`.
#
# ### Keyword arguments:
# Keyword arguments and their default values are given by:
# *   `krylovdim = 30`: the maximum dimension of the Krylov subspace that will be constructed.
#     Note that the dimension of the vector space is not known or checked, e.g. `x₀` should
#     not necessarily support the `Base.length` function. If you know the actual problem
#     dimension is smaller than the default value, it is useful to reduce the value of
#     `krylovdim`, though in principle this should be detected.
# *   `tol = 1e-12`: the requested accuracy (corresponding to the 2-norm of the residual for
#     Schur vectors, not the eigenvectors). If you work in e.g. single precision (`Float32`),
#     you should definitely change the default value.
# *   `maxiter::Int = 100`: the number of times the Krylov subspace can be rebuilt; see below
#     for further details on the algorithms.
# *   `info::Int = 0`: the level of verbosity, default is zero (no output)
# *   `issymmetric`: if the linear map is symmetric, only meaningful if `T<:Real`
# *   `ishermitian`: if the linear map is hermitian
# The default value for the last two depends on the method. If an `AbstractMatrix` is used,
# `issymmetric` and `ishermitian` are checked for that matrix, ortherwise the default values
# are `issymmetric = false` and `ishermitian = T <: Real && issymmetric`.
#
# ### Algorithm
# The last method, without default values and keyword arguments, is the one that is finally
# called, and can also be used directly. Here, one specifies the algorithm explicitly as
# either [`Lanczos`](@ref), for real symmetric or complex hermitian problems, or
# [`Arnoldi`](@ref), for general problems. Note that these names refer to the process for
# building the Krylov subspace.
#
# !!! warning "`Arnoldi` not yet implented"
# """
function geometricseries end

function geometricseries(A, b, x₀=rmul!(similar(b), false); kwargs...)
    alg = eigselector(A, promote_type(eltype(b), eltype(x₀)); kwargs...)
    return geometricseries(A, b, x₀, alg)
end

function geometricseries(A, b, x₀, alg::Lanczos)
    # Initial function operation and division defines number type
    y₀ = apply(A, x₀)
    numops = 1
    T = typeof(dot(b, y₀) / norm(b))

    r = copyto!(similar(b, T), b)
    r = axpy!(-1, x₀, r)
    r = axpy!(+1, y₀, r)
    β = norm(r)
    S = typeof(β)
    tol::S = alg.tol
    x = copyto!(similar(r), x₀)

    # krylovdim and related allocations
    krylovdim = alg.krylovdim
    PP1 = Matrix{S}(undef, (krylovdim, krylovdim))
    PP2 = Matrix{S}(undef, (krylovdim, krylovdim))
    GG = Matrix{S}(undef, (krylovdim, krylovdim))

    # initialize iterator
    iter = LanczosIterator(A, r, alg.orth)
    fact = initialize(iter; info=alg.info - 2)
    numops += 1
    sizehint!(fact, krylovdim)
    maxiter = alg.maxiter
    numiter = 0
    while true
        while normres(fact) > tol && length(fact) < krylovdim
            fact = expand!(iter, fact; info=alg.info - 2)
            numops += 1
        end
        K = fact.k # current Krylov dimension
        V = basis(fact)

        # Small matrix exponential and error estimation
        H = rayleighquotient(fact)
        P1 = copyto!(view(PP1, 1:K, 1:K), rayleighquotient(fact))
        P2 = copyto!(view(PP2, 1:K, 1:K), rayleighquotient(fact))
        GG = copyto!(view(GG, 1:K, 1:K), I)

        while (GG[K, 1] + P1[K, 1]) * normres(fact) < tol
            GG .+= P1
            mul!(P2, P1, H)
            P1, P2 = P2, P1
        end
        unproject!(x, V, view(GG, :, 1), β, 1)
        r = apply(A, x)
        numops += 1
        r = axpy!(+1, b, r)
        r = axpy!(-1, x, r)
        β = norm(r)
        numiter += 1

        if alg.info > 1
            @info "Geometric series in iteration $numiter: norm residual $β"
        end
        if β <= tol
            if alg.info > 0
                @info "Geometric series finished in iteration $numiter: norm residual $β"
            end
            return x, ConvergenceInfo(1, r, β, numiter, numops)
        end
        if numiter == maxiter
            if alg.info > 0
                @warn "Geometric series finished without convergence after $numiter iterations"
            end
            return x, ConvergenceInfo(0, r, β, numiter, numops)
        end
        r = rmul!(r, 1 / β)
        iter = LanczosIterator(A, r, alg.orth)
        fact = initialize!(iter, fact; info=alg.info - 2)
        numops += 1
    end
end

function geometricseries(A, b, x₀, alg::Arnoldi)
    # Initial function operation and division defines number type
    y₀ = apply(A, x₀)
    numops = 1
    T = typeof(dot(b, y₀) / norm(b))

    r = copyto!(similar(b, T), b)
    r = axpy!(-1, x₀, r)
    r = axpy!(+1, y₀, r)
    β = norm(r)
    tol::typeof(β) = alg.tol
    x = copyto!(similar(r), x₀)

    # krylovdim and related allocations
    krylovdim = alg.krylovdim
    HH = Matrix{T}(undef, (krylovdim, krylovdim))
    PP1 = Matrix{T}(undef, (krylovdim, krylovdim))
    PP2 = Matrix{T}(undef, (krylovdim, krylovdim))
    GG = Matrix{T}(undef, (krylovdim, krylovdim))

    # initialize iterator
    iter = ArnoldiIterator(A, r, alg.orth)
    fact = initialize(iter; info=alg.info - 2)
    numops += 1
    sizehint!(fact, krylovdim)
    maxiter = alg.maxiter
    numiter = 0
    while true
        while normres(fact) > tol && length(fact) < krylovdim
            fact = expand!(iter, fact; info=alg.info - 2)
            numops += 1
        end
        K = fact.k # current Krylov dimension
        V = basis(fact)

        # Small matrix exponential and error estimation
        H = copyto!(view(HH, 1:K, 1:K), rayleighquotient(fact))
        P1 = copyto!(view(PP1, 1:K, 1:K), rayleighquotient(fact))
        P2 = copyto!(view(PP2, 1:K, 1:K), rayleighquotient(fact))
        GG = copyto!(view(GG, 1:K, 1:K), I)

        while (GG[K, 1] + P1[K, 1]) * normres(fact) < tol
            GG .+= P1
            mul!(P2, P1, H)
            P1, P2 = P2, P1
        end
        unproject!(x, V, view(GG, :, 1), β, 1)
        r = apply(A, x)
        numops += 1
        r = axpy!(+1, b, r)
        r = axpy!(-1, x, r)
        β = norm(r)
        numiter += 1

        if alg.info > 1
            @info "Geometric series in iteration $numiter: norm residual $β"
        end
        if β <= tol
            if alg.info > 0
                @info "Geometric series finished in iteration $numiter: norm residual $β"
            end
            return x, ConvergenceInfo(1, r, β, numiter, numops)
        end
        if numiter == maxiter
            if alg.info > 0
                @warn "Geometric series finished without convergence after $numiter iterations"
            end
            return x, ConvergenceInfo(0, r, β, numiter, numops)
        end
        r = rmul!(r, 1 / β)
        iter = ArnoldiIterator(A, r, alg.orth)
        fact = initialize!(iter, fact; info=alg.info - 2)
        numops += 1
    end
end
