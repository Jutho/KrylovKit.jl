"""
    initialize_vector(f, A)

Construct a starting vector for the Krylov subspace of the operator `A` constructed for `f`.
For `A::AbstractMatrix`, the default is a random vector of the same size as the number of rows of `A`.
For a function `A` or custom object, you should either provide a starting vector `x₀` or implement this function.
"""
function initialize_vector(f, A::AbstractMatrix)
    return Random.rand!(similar(A, scalartype(A), size(A, 1)))
end

function initialize_vector(f, A)
    error("""
          Cannot construct a starting vector for the Krylov subspace from the given operator.
          Either provide a starting vector `x₀`, or implement [`initialize_vector`](@ref) for values of type `$(typeof(f)), $(typeof(A))`.
          """)
    return nothing
end
