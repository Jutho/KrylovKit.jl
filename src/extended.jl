
# function factorize(f, x::Vector, factorizer::Arnoldi)
#     # Based on Morgan, Mathematics of Computation 65, 1213 (1996)
#     n = size(x,1)
#     k = size(x,2)
#     m = factorizer.krylovdim
#     p = m - k
#
#     η = factorizer.tol
#
#     @inbounds begin
#         q = normalize!(x[:,1]);
#         fq = f(q)
#         h = dot(q, fq)
#
#         # Define data structures to store factorization
#         Q = OrthonormalBasis{eltype(fq)}(n)
#         sizehint!(Q, m)
#         H = zeros(typeof(h), m, m)
#
#         # Initialize
#         push!(Q, q)
#         H[1,1] = h
#         Base.LinAlg.axpy!(-h,q,fq);
#         β = norm(fq)
#         j = 1
#
#         # While vectors in x define invariant subspace, keep adding
#         while β <= η && j < k
#             j += 1
#             q, = orthonormalize!(x[:,j], Q)
#             push!(Q, q)
#             fq = f(q)
#             orthogonalize!(fq, Q, slice(H,1:j,j))
#             β = norm(fq)
#         end
#
#         # Arnoldi loop for creating new vectors
#         k = k - j # how many vectors are there still left in x
#         p = m - k
#         while β > η && j < p
#             H[j+1,j] = β
#             q = scale!(fq, 1/β)
#             j += 1
#             push!(Q, q)
#             fq = f(q)
#             orthogonalize!(fq, Q, slice(H,1:j,j))
#             β = norm(fq)
#         end
#         p = j  # Actual last Krylov vector
#
#         # Define data structure for storing residual vectors
#         R = Vector{typeof(fq)}(k+1)
#         R[1] = fq
#
#         # Add extra vectors from x
#         for i = 1:k
#             q, = orthonormalize!(x[:,end-k+i], Q)
#             j = j+1
#             push!(Q, q)
#             H[j,p] = dot(q,R[1])
#             Base.LinAlg.axpy!(-H[j,p],q,R[1]);
#         end
#         m = j # Actual dimension of the subspace
#
#         # Compute final columns of projected matrix
#         j = p+1
#         for i = 1:k
#             q = Q[j]
#             fq = f(q)
#             orthogonalize!(fq, Q, slice(H,1:m,j))
#             R[i+1] = fq
#             j += 1
#         end
#
#         # Shrink to actual size if necessary
#         if size(H,1) != m
#             H = H[1:m,1:m]
#         end
#     end
#
#     return ArnoldiFact(Q,H,R)
# end
