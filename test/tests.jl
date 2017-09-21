using KrylovKit


for T in (Float32, Float64, Complex64, Complex128)
    for alg in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
        A = rand(T,(100,100))
        v = rand(T,(100,))
        iter = arnoldi(A, v, alg)
        s = start(iter)
        while !done(iter, s)
            fact, s = next(iter, s)
        end

        A = (A+A')
        iter = lanczos(A, v, alg)
        s = start(iter)
        while !done(iter, s)
            fact, s = next(iter, s)
        end
    end
end


function cpmap(A,ρ)
    D1,d,D2 = size(A)
    σ = ρ-reshape(reshape(A,D1*d, D2)*ρ,D1,d*D2)*reshape(A,D1,d*D2)'
    return σ
end

D = 100
d = 2
A = reshape(qr(randn(D*d,D))[1], (D,d,D))

X = randn(D,D)
ρ₀ = X*X'
scale!(ρ₀, 1/vecnorm(ρ₀))

alg = GMRES(krylovdim = 30, tol = 1e-12)
linsolve(x->cpmap(A,x), cpmap(A, ρ₀), alg)



x0 = zeros(10)
A = randn(10,10)
A = (A*A')
b = randn(10)

r0 = b-A*x0

K = 11

x1=Vector{Vector{Float64}}(K)
r1=Vector{Vector{Float64}}(K)
p=Vector{Vector{Float64}}(K)
a=Vector{Float64}(K)
b=Vector{Float64}(K)

x2=Vector{Vector{Float64}}(K)
r2=Vector{Vector{Float64}}(K)
v=Vector{Vector{Float64}}(K)
w=Vector{Vector{Float64}}(K)
α=Vector{Float64}(K)
β=Vector{Float64}(K)
γ=Vector{Float64}(K)
δ=Vector{Float64}(K)
q=Vector{Float64}(K)

x1[1] = x0
r1[1] = r0
p[1] = r0

for k = 1:K-1
    z = A*p[k]
    a[k] = dot(r1[k],r1[k])/dot(p[k],z)
    x1[k+1] = x1[k] + a[k] * p[k]
    r1[k+1] = r1[k] - a[k] * z
    b[k] = dot(r1[k+1],r1[k+1])/dot(r1[k],r1[k])
    p[k+1] = r1[k+1] + b[k] * p[k]
end

x2[1] = x0
r2[1] = r0
v[1] = r0/norm(r0)
w[1] = r0/norm(r0)
for k = 1:K-1
    z = A*v[k]
    α[k] = dot(v[k],z)
    z -= α[k]*v[k]
    if k > 1
        z -= β[k-1]*v[k-1]
    end
    δ[k] = k==1 ? α[k] : α[k] - γ[k-1]*γ[k-1]*δ[k-1]
    q[k] = k==1 ? vecnorm(r0)/δ[k] : -γ[k-1]*δ[k-1]*q[k-1]/δ[k]

    x2[k+1] = x2[k] + q[k]*w[k]
    r2[k+1] = -q[k]*z

    β[k] = norm(z)
    v[k+1] = z/β[k]

    γ[k] = β[k]/δ[k]
    w[k+1] = v[k+1] - γ[k]*w[k]
end
