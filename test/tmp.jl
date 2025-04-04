using LinearAlgebra
function compute_residual!(R::AbstractVector{T}, A_X::AbstractVector{T}, X::AbstractVector{T}, M::AbstractMatrix, X_prev::AbstractVector{T}, B_prev::AbstractMatrix) where T
    @inbounds for j in 1:length(X)
        r_j = R[j] 
        copyto!(r_j, A_X[j])
        @simd for i in 1:length(X)
            axpy!(- M[i,j], X[i], r_j)
        end
        @simd for i in 1:length(X_prev)
            axpy!(- B_prev[i,j], X_prev[i], r_j)
        end
    end
    return R
end
n=1000000
m=10
AX = [rand(n) for i in 1:m];
X = [rand(n) for i in 1:m];
M = rand(m,m);
X_prev = [rand(n) for i in 1:m];
B_prev = rand(m,m);
R = [rand(n) for i in 1:m];
compute_residual!(R, AX, X, M, X_prev, B_prev);

AX_h = hcat(AX...);
X_h = hcat(X...);
X_prev_h = hcat(X_prev...);
R_h = AX_h - X_h * M - X_prev_h * B_prev;

norm(R_h - hcat(R...))


using LinearAlgebra
x = [rand(5) for _ in 1:3];
y = [rand(5) for _ in 1:3];
x
y
y = deepcopy(x)
y[1][1] =1.0
y
x