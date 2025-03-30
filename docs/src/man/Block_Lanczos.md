# General Block Lanczos
$A$ is a Hermitian matrix, 
$$A = Q T Q'$$

$T$ is a tridiagonal matrix, 
$$T = \begin{pmatrix}
    \alpha_1 & \beta_1 \\
    \beta_1 & \alpha_2 & \beta_2 \\
    & \beta_2 & \ddots & \ddots \\
    && \ddots & \alpha_{n-1} & \beta_{n-1} \\
    &&& \beta_{n-1} & \alpha_n
\end{pmatrix}$$

But $\beta \neq 0$ and thus eigenvalues of $T$ are different from each other. 

To get multiple eigenvalues, we can use block Lanczos. 

$$A = Q T Q', \quad Q = [X_1|..|X_n],\quad \text{size}(X_i) = (p,p)$$

$$T = \begin{pmatrix}
    M_1 & B_1'\\
    B_1 & M_2 & B_2'\\
    & B_2 & \ddots & \ddots \\
    && \ddots & M_{n-1} & B_{n-1}'\\
    &&& B_{n-1} & M_n
\end{pmatrix}$$

It's advantage is to get multiple eigenvalues. But it's disadvantage is that it can cause ghost eigenvalues because of the loss of orthogonality of the Lanczos vectors. 

So my solution is to make sure each $X_i$ we get is orthogonal to $X_{i_1},..,X_1$:

$$X_i = X_i - Q_{i-1}*(Q_{i-1}'X_i)\\
Q_{i-1} = [X_1|..|X_{i-1}]
$$

I use Modified Schidi's method to force the orthogonality of the basis And use some skills to improve the method and speed up. 

# Difference between Block Lanczos and Lanczos in code

1. I don't use inner because it maps a couple of matrix to a scalar. 
2. I have add test to make sure Block Lanczos can work for map input
3. I add SaprseArray to do test for sparse matrix input

