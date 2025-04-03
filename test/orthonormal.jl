@testset "abstract_qr" begin
    @testset "Dense matrix tests" begin
        n = 10
        k = 5
        vecs = [randn(n) for _ in 1:k]
        A = hcat(vecs...)
        B = OrthonormalBasis{Vector{Float64}}(vecs)
        
        for orth in [ClassicalGramSchmidt(), ModifiedGramSchmidt(), 
                     ClassicalGramSchmidt2(), ModifiedGramSchmidt2(),
                     ClassicalGramSchmidtIR(0.7), ModifiedGramSchmidtIR(0.7)]
            Q, R = abstract_qr(B, alg=orth)
            
            for i in 1:length(Q), j in 1:length(Q)
                expected = i == j ? 1.0 : 0.0
                @test abs(inner(Q[i], Q[j]) - expected) < 1e-10
            end
            
            for i in 2:size(R, 1), j in 1:i-1
                @test abs(R[i, j]) < 1e-10
            end
            
            reconstructed = zeros(n, k)
            for j in 1:k
                for i in 1:length(Q)
                    reconstructed[:, j] .+= Q[i] .* R[i, j]
                end
            end
            @test norm(reconstructed - A) / norm(A) < 1e-10
        end
    end
    
    @testset "Linearly dependent vectors" begin
        n = 10
        vecs = [randn(n) for _ in 1:4]
        push!(vecs, vecs[1] + vecs[2])
        B = OrthonormalBasis{Vector{Float64}}(vecs)
        
        Q, R = abstract_qr(B)
        
        singular_values = svdvals(R)
        rank_R = count(s -> s > 1e-10, singular_values)
        @test rank_R == 4
        
        for i in 1:length(Q), j in 1:length(Q)
            expected = i == j ? 1.0 : 0.0
            @test abs(inner(Q[i], Q[j]) - expected) < 1e-10
        end
    end
    
    @testset "Complex vectors" begin
        n = 8
        k = 4
        vecs = [randn(ComplexF64, n) for _ in 1:k]
        A = hcat(vecs...)
        B = OrthonormalBasis{Vector{ComplexF64}}(vecs)
        
        Q, R = abstract_qr(B)
        
        for i in 1:length(Q), j in 1:length(Q)
            expected = i == j ? 1.0 : 0.0
            @test abs(inner(Q[i], Q[j]) - expected) < 1e-10
        end
        
        reconstructed = zeros(ComplexF64, n, k)
        for j in 1:k
            for i in 1:length(Q)
                reconstructed[:, j] .+= Q[i] .* R[i, j]
            end
        end
        @test norm(reconstructed - A) / norm(A) < 1e-10
    end
    
    @testset "Custom vector types" begin
        struct MyVector{T}
            data::Vector{T}
        end
        
        Base.similar(v::MyVector) = MyVector(similar(v.data))
        Base.copyto!(dest::MyVector, src::MyVector) = (copyto!(dest.data, src.data); dest)
        KrylovKit.inner(x::MyVector, y::MyVector) = dot(x.data, y.data)
        Base.length(v::MyVector) = length(v.data)
        Base.:*(v::MyVector, α::Number) = MyVector(v.data * α)
        Base.getindex(v::MyVector, i) = v.data[i]
        Base.setindex!(v::MyVector, val, i) = (v.data[i] = val; v)
        LinearAlgebra.norm(v::MyVector) = norm(v.data)
        
        n = 5
        k = 3
        vecs = [MyVector(randn(n)) for _ in 1:k]
        B = OrthonormalBasis{MyVector{Float64}}(vecs)
        
        try
            Q, R = abstract_qr(B)
            @test length(Q) == k
            @test size(R) == (k, k)
        catch e
            @test false
        end
    end
end


using KrylovKit,LinearAlgebra
n = 10
vecs = [randn(n) for _ in 1:4]
B = KrylovKit.OrthonormalBasis{Vector{Float64}}(vecs)

_,R = KrylovKit.abstract_qr!(B; tol=1e-10)
R
C = B.basis
C'
D = zeros(length(C),length(C))
for i in 1:length(C)
    for j in 1:length(C)
        D[i,j] = C[i]'*C[j]
    end
end
D

using LinearAlgebra
function my_f!(A)
    Aq,B = qr(A)
    A .= Matrix(Aq)
end
A = [1.0 2 ;3 4]
my_f!(A)
A



using LinearAlgebra
function LinearAlgebra.mul!(A::AbstractVector{T},B::AbstractVector{T},M::AbstractMatrix) where T
    @inbounds for i in eachindex(A)
        @simd for j in eachindex(B)
            A[i] += M[j,i] * B[j]
        end
    end
    return A
end
A = [rand(4) for _ in 1:2]
Ac = copy(A)
B = [rand(4) for _ in 1:2]
M = rand(2,2)
mul!(A,B,M)
hcat(Ac...) + hcat(B...) * M
A