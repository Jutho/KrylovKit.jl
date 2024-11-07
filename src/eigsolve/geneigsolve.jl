"""
    geneigsolve((A::AbstractMatrix, B::AbstractMatrix), [howmany = 1, which = :LM,
                                    T = promote_type(eltype(A), eltype(B))]; kwargs...)
    geneigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)
    geneigsolve(f, x₀, [howmany = 1, which = :LM]; kwargs...)
    # expert version:
    geneigsolve(f, x₀, howmany, which, algorithm)

Compute at least `howmany` generalized eigenvalues ``λ`` and generalized eigenvectors
``x`` of the form ``(A - λB)x = 0``, where `A` and `B` are either instances of
`AbstractMatrix`, or some function that implements the matrix vector product. In case
functions are used, one could either specify the action of `A` and `B` via a tuple of two
functions (or a function and an `AbstractMatrix`), or one could use a single function that
takes a single argument `x` and returns two results, corresponding to `A*x` and `B*x`.
Return the computed eigenvalues, eigenvectors and a `ConvergenceInfo` structure.

### Arguments:

The first argument is either a tuple of two linear maps, so a function or an `AbstractMatrix`
for either of them, representing the action of `A` and `B`. Alternatively, a single function
can be used that takes a single argument `x` and returns the equivalent of `(A*x, B*x)` as
result. This latter form is compatible with the `do` block syntax of Julia. If an
`AbstractMatrix` is used for either `A` or `B`, a starting vector `x₀` does not need to be
provided, it is then chosen as `rand(T, size(A,1))` if `A` is an `AbstractMatrix` (or
similarly if only `B` is an `AbstractMatrix`). Here `T = promote_type(eltype(A), eltype(B))`
if both `A` and `B` are instances of `AbstractMatrix`, or just the `eltype` of whichever is
an `AbstractMatrix`. If both `A` and `B` are encoded more generally as a callable function
or method, the best approach is to provide an explicit starting guess `x₀`. Note that `x₀`
does not need to be of type `AbstractVector`, any type that behaves as a vector and supports
the required methods (see KrylovKit docs) is accepted. If instead of `x₀` an integer `n` is
specified, it is assumed that `x₀` is a regular vector and it is initialized to `rand(T,n)`,
where the default value of `T` is `Float64`, unless specified differently.

The next arguments are optional, but should typically be specified. `howmany` specifies how
many eigenvalues should be computed; `which` specifies which eigenvalues should be
targeted. Valid specifications of `which` are given by

  - `:LM`: eigenvalues of largest magnitude
  - `:LR`: eigenvalues with largest (most positive) real part
  - `:SR`: eigenvalues with smallest (most negative) real part
  - `:LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
  - `:SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
  - [`EigSorter(f; rev = false)`](@ref): eigenvalues `λ` that appear first (or last if
    `rev == true`) when sorted by `f(λ)`

!!! note "Note about selecting `which` eigenvalues"

    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. Even with `ClosestTo`, no shift and invert is performed.
    This is useful if, e.g., you know the spectrum to be within the unit circle in the
    complex plane, and want to target the eigenvalues closest to the value `λ = 1`.

The argument `T` acts as a hint in which `Number` type the computation should be performed,
but is not restrictive. If the linear map automatically produces complex values, complex
arithmetic will be used even though `T<:Real` was specified.

### Return values:

The return value is always of the form `vals, vecs, info = geneigsolve(...)` with

  - `vals`: a `Vector` containing the eigenvalues, of length at least `howmany`, but could
    be longer if more eigenvalues were converged at the same cost.

  - `vecs`: a `Vector` of corresponding eigenvectors, of the same length as `vals`.
    Note that eigenvectors are not returned as a matrix, as the linear map could act on any
    custom Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`, up to a possibly
    different `eltype`. When the linear map is a simple `AbstractMatrix`, `vecs` will be
    `Vector{Vector{<:Number}}`.
  - `info`: an object of type [`ConvergenceInfo`], which has the following fields

      + `info.converged::Int`: indicates how many eigenvalues and eigenvectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
      + `info.residual::Vector`: a list of the same length as `vals` containing the
        residuals `info.residual[i] = f(vecs[i]) - vals[i] * vecs[i]`
      + `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
      + `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
      + `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)

!!! warning "Check for convergence"

    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Keyword arguments:

Keyword arguments and their default values are given by:

  - `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
  - `tol::Real`: the requested accuracy, relative to the 2-norm of the corresponding
    eigenvectors, i.e. convergence is achieved if `norm((A - λB)x) < tol * norm(x)`. Because
    eigenvectors are now normalised such that `dot(x, B*x) = 1`, `norm(x)` is not
    automatically one. If you work in e.g. single precision (`Float32`), you should
    definitely change the default value.
  - `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed. Note that the dimension of the vector space is not known or checked, e.g.
    `x₀` should not necessarily support the `Base.length` function. If you know the actual
    problem dimension is smaller than the default value, it is useful to reduce the value
    of `krylovdim`, though in principle this should be detected.
  - `maxiter::Integer`: the number of times the Krylov subspace can be rebuilt; see below
    for further details on the algorithms.
  - `orth::Orthogonalizer`: the orthogonalization method to be used, see
    [`Orthogonalizer`](@ref)
  - `issymmetric::Bool`: if both linear maps `A` and `B` are symmetric, only meaningful if
    `T<:Real`
  - `ishermitian::Bool`: if both linear maps `A` and `B` are hermitian
  - `isposdef::Bool`: if the linear map `B` is positive definite
    
The default values are given by `tol = KrylovDefaults.tol`,
`krylovdim = KrylovDefaults.krylovdim`, `maxiter = KrylovDefaults.maxiter`,
`orth = KrylovDefaults.orth`; see [`KrylovDefaults`](@ref) for details.

The default value for the last three parameters depends on the method. If an
`AbstractMatrix` is used, `issymmetric`, `ishermitian` and `isposdef` are checked for that
matrix, otherwise the default values are `issymmetric = false` and
`ishermitian = T <: Real && issymmetric`. When values are provided, no checks will be
performed even in the matrix case.

### Algorithm

The last method, without default values and keyword arguments, is the one that is finally
called, and can also be used directly. Here the algorithm is specified, though currently
only [`GolubYe`](@ref) is available. The Golub-Ye algorithm is an algorithm for solving
hermitian (symmetric) generalized eigenvalue problems `A x = λ B x` with positive definite
`B`, without the need for inverting `B`. It builds a Krylov subspace of size `krylovdim`
starting from an estimate `x` by acting with `(A - ρ(x) B)`, where 
`ρ(x) = dot(x, A*x)/ dot(x, B*x)`, and employing the Lanczos algorithm. This process is
repeated at most `maxiter` times. In every iteration `k>1`, the subspace will also be
expanded to size `krylovdim+1` by adding ``x_k - x_{k-1}``, which is known as the LOPCG
correction and was suggested by Money and Ye. With `krylovdim = 2`, this algorithm becomes
equivalent to `LOPCG`.

!!! warning "Restriction to symmetric definite generalized eigenvalue problems"

    While the only algorithm so far is restricted to symmetric/hermitian generalized
    eigenvalue problems with positive definite `B`, this is not reflected in the default
    values for the keyword arguments `issymmetric` or `ishermitian` and `isposdef`. Make
    sure to set these to true to understand the implications of using this algorithm.
"""
function geneigsolve end

function geneigsolve(AB::Tuple{AbstractMatrix,AbstractMatrix},
                     howmany::Int=1,
                     which::Selector=:LM,
                     T=promote_type(eltype.(AB)...);
                     kwargs...)
    if !(size(AB[1], 1) == size(AB[1], 2) == size(AB[2], 1) == size(AB[2], 2))
        throw(DimensionMismatch("Matrices `A` and `B` should be square and have matching size"))
    end
    x₀ = Random.rand!(similar(AB[1], T, size(AB[1], 1)))
    return geneigsolve(AB, x₀, howmany::Int, which; kwargs...)
end
function geneigsolve(AB::Tuple{Any,AbstractMatrix},
                     howmany::Int=1,
                     which::Selector=:LM,
                     T=eltype(AB[2]);
                     kwargs...)
    x₀ = Random.rand!(similar(AB[2], T, size(AB[2], 1)))
    return geneigsolve(AB, x₀, howmany, which; kwargs...)
end
function geneigsolve(AB::Tuple{AbstractMatrix,Any},
                     howmany::Int=1,
                     which::Selector=:LM,
                     T=eltype(AB[1]);
                     kwargs...)
    x₀ = Random.rand!(similar(AB[1], T, size(AB[1], 1)))
    return geneigsolve(AB, x₀, howmany, which; kwargs...)
end

function geneigsolve(f,
                     n::Int,
                     howmany::Int=1,
                     which::Selector=:LM,
                     T::Type=Float64;
                     kwargs...)
    return geneigsolve(f, rand(T, n), howmany, which; kwargs...)
end

function geneigsolve(f, x₀, howmany::Int=1, which::Selector=:LM; kwargs...)
    Tx = typeof(x₀)
    Tfx = Core.Compiler.return_type(genapply, Tuple{typeof(f),Tx}) # should be a tuple type
    Tfx1 = Base.tuple_type_head(Tfx)
    Tfx2 = Base.tuple_type_head(Base.tuple_type_tail(Tfx))
    T1 = Core.Compiler.return_type(dot, Tuple{Tx,Tfx1})
    T2 = Core.Compiler.return_type(dot, Tuple{Tx,Tfx2})
    T = promote_type(T1, T2)
    alg = geneigselector(f, T; kwargs...)
    if alg isa GolubYe && (which == :LI || which == :SI)
        error("Eigenvalue selector which = $which invalid: real eigenvalues expected with Lanczos algorithm")
    end
    return geneigsolve(f, x₀, howmany, which, alg)
end

function geneigselector(AB::Tuple{AbstractMatrix,AbstractMatrix},
                        T::Type;
                        issymmetric=T <: Real && all(LinearAlgebra.issymmetric, AB),
                        ishermitian=issymmetric || all(LinearAlgebra.ishermitian, AB),
                        isposdef=ishermitian && LinearAlgebra.isposdef(AB[2]),
                        kwargs...)
    if (issymmetric || ishermitian) && isposdef
        return GolubYe(; kwargs...)
    else
        throw(ArgumentError("Only symmetric or hermitian generalized eigenvalue problems with positive definite `B` matrix are currently supported."))
    end
end
function geneigselector(f,
                        T::Type;
                        issymmetric=false,
                        ishermitian=issymmetric,
                        isposdef=false,
                        kwargs...)
    if (issymmetric || ishermitian) && isposdef
        return GolubYe(; kwargs...)
    else
        throw(ArgumentError("Only symmetric or hermitian generalized eigenvalue problems with positive definite `B` matrix are currently supported."))
    end
end
