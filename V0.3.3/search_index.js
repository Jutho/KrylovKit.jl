var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#KrylovKit.jl-1",
    "page": "Home",
    "title": "KrylovKit.jl",
    "category": "section",
    "text": "A Julia package collecting a number of Krylov-based algorithms for linear problems, singular value and eigenvalue problems and the application of functions of linear maps or operators to vectors."
},

{
    "location": "#Overview-1",
    "page": "Home",
    "title": "Overview",
    "category": "section",
    "text": "KrylovKit.jl accepts general functions or callable objects as linear maps, and general Julia objects with vector like behavior (see below) as vectors.The high level interface of KrylovKit is provided by the following functions:linsolve: solve linear systems A*x = b\neigsolve: find a few eigenvalues and corresponding eigenvectors of an   eigenvalue problem A*x = λ x\ngeneigsolve: find a few eigenvalues and corresponding vectors of a   generalized eigenvalue problem A*x = λ*B*x\nsvdsolve: find a few singular values and corresponding left and right   singular vectors A*x = σ * y and A\'*y = σ*x.\nexponentiate: apply the exponential of a linear map to a vector\nexpintegrator: exponential integrator for a linear non-homogeneous ODE,   generalization of exponentiate"
},

{
    "location": "#Package-features-and-alternatives-1",
    "page": "Home",
    "title": "Package features and alternatives",
    "category": "section",
    "text": "This section could also be titled \"Why did I create KrylovKit.jl\"?There are already a fair number of packages with Krylov-based or other iterative methods, such asIterativeSolvers.jl: part of the   JuliaMath organisation, solves linear systems and least   square problems, eigenvalue and singular value problems\nKrylov.jl: part of the   JuliaSmoothOptimizers organisation, solves   linear systems and least square problems, specific for linear operators from   LinearOperators.jl.\nKrylovMethods.jl: specific for sparse   matrices\nExpokit.jl: application of the matrix   exponential to a vector\nArnoldiMethod.jl: Implicitly restarted   Arnoldi method for eigenvalues of a general matrix\nJacobiDavidson.jl: Jacobi-Davidson   method for eigenvalues of a general matrixThese packages have certainly inspired and influenced the development of KrylovKit.jl. However, KrylovKit.jl distinguishes itself from the previous packages in the following ways:KrylovKit accepts general functions to represent the linear map or operator that defines  the problem, without having to wrap them in a  LinearMap or  LinearOperator type.  Of course, subtypes of AbstractMatrix are also supported. If the linear map (always  the first argument) is a subtype of AbstractMatrix, matrix vector multiplication is  used, otherwise it is applied as a function call.\nKrylovKit does not assume that the vectors involved in the problem are actual subtypes  of AbstractVector. Any Julia object that behaves as a vector is supported, so in  particular higher-dimensional arrays or any custom user type that supports the  following functions (with v and w two instances of this type and α, β scalars  (i.e. Number)):\nBase.eltype(v): the scalar type (i.e. <:Number) of the data in v\nBase.similar(v, [T::Type<:Number]): a way to construct additional similar vectors,   possibly with a different scalar type T.\nBase.copyto!(w, v): copy the contents of v to a preallocated vector w\nLinearAlgebra.mul!(w, v, α): out of place scalar multiplication; multiply   vector v with scalar α and store the result in w\nLinearAlgebra.rmul!(v, α): in-place scalar multiplication of v with α; in   particular with α = false, v is initialized with all zeros\nLinearAlgebra.axpy!(α, v, w): store in w the result of α*v + w\nLinearAlgebra.axpby!(α, v, β, w): store in w the result of α*v + β*w\nLinearAlgebra.dot(v,w): compute the inner product of two vectors\nLinearAlgebra.norm(v): compute the 2-norm of a vector\nFurthermore, KrylovKit provides two types satisfying the above requirements that might  facilitate certain applications:\nRecursiveVec can be used for grouping a set of vectors into a single   vector like structure (can be used recursively). The reason that e.g.   Vector{<:Vector} cannot be used for this is that it returns the wrong eltype   and methods like similar(v, T) and fill!(v, α)   don\'t work correctly.\nInnerProductVec can be used to redefine the inner product (i.e. dot)   and corresponding norm (norm) of an already existing vector like object. The   latter should help with implementing certain type of preconditioners"
},

{
    "location": "#Current-functionality-1",
    "page": "Home",
    "title": "Current functionality",
    "category": "section",
    "text": "The following algorithms are currently implementedlinsolve: CG, GMRES\neigsolve: a Krylov-Schur algorithm (i.e. with tick restarts) for extremal eigenvalues   of normal (i.e. not generalized) eigenvalue problems, corresponding to   Lanczos for real symmetric or complex hermitian linear maps, and to   Arnoldi for general linear maps.\ngeneigsolve: an customized implementation of the inverse-free algorithm of Golub and   Ye for symmetric / hermitian generalized eigenvalue problems with positive definite   matrix B in the right hand side of the generalized eigenvalue problem A v = B v λ.   The Matlab implementation was described by Money and Ye and is known as EIGIFP; in   particular it extends the Krylov subspace with a vector corresponding to the step   between the current and previous estimate, analoguous to the locally optimal   preconditioned conjugate gradient method (LOPCG). In particular, with Krylov dimension   2, it becomes equivalent to the latter.\nsvdsolve: finding largest singular values based on Golub-Kahan-Lanczos   bidiagonalization (see GKL)\nexponentiate: a Lanczos based algorithm for the action of the exponential of   a real symmetric or complex hermitian linear map.\nexpintegrator: exponential integrator for a linear non-homogeneous ODE, computes a   linear combination of the so-called ϕⱼ functions which generalize ϕ₀(z) = exp(z)."
},

{
    "location": "#Future-functionality?-1",
    "page": "Home",
    "title": "Future functionality?",
    "category": "section",
    "text": "Here follows a wish list / to-do list for the future. Any help is welcomed and appreciated.More algorithms, including biorthogonal methods:\nfor linsolve: MINRES, BiCG, BiCGStab(l), IDR(s), ...\nfor eigsolve: BiLanczos, Jacobi-Davidson JDQR/JDQZ, subspace iteration (?), ...\nfor geneigsolve: trace minimization, block versions\nSupport both in-place / mutating and out-of-place functions as linear maps\nLeast square problems\nNonlinear eigenvalue problems\nPreconditioners\nRefined Ritz vectors, Harmonic ritz values and vectors\nReuse memory for storing vectors when restarting algorithms\nBlock versions of the algorithms\nMore relevant matrix functionsPartially done:Improved efficiency for the specific case where x is Vector (i.e. BLAS level 2   operations): any vector v::AbstractArray which has IndexStyle(v) == IndexLinear()   now benefits from a multithreaded (use export JULIA_NUM_THREADS = x with x the   number of threads you want to use) implementation that resembles BLAS level 2 style for   the vector operations, provided ClassicalGramSchmidt(), ClassicalGramSchmidt2() or   ClassicalGramSchmidtIR() is chosen as orthogonalization routine."
},

{
    "location": "man/intro/#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "man/intro/#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "Pages = [\"man/intro.md\", \"man/linear.md\", \"man/eig.md\", \"man/svd.md\", \"man/matfun.md\",\n\"man/algorithms.md\", \"man/implementation.md\"]\nDepth = 2"
},

{
    "location": "man/intro/#Installing-1",
    "page": "Introduction",
    "title": "Installing",
    "category": "section",
    "text": "Install KrylovKit.jl via the package manager:using Pkg\nPkg.add(\"KrylovKit\")KrylovKit.jl is a pure Julia package; no dependencies (aside from the Julia standard library) are required."
},

{
    "location": "man/intro/#KrylovKit",
    "page": "Introduction",
    "title": "KrylovKit",
    "category": "module",
    "text": "KrylovKit\n\nA Julia package collecting a number of Krylov-based algorithms for linear problems, singular value and eigenvalue problems and the application of functions of linear maps or operators to vectors.\n\nKrylovKit accepts general functions or callable objects as linear maps, and general Julia objects with vector like behavior as vectors.\n\nThe high level interface of KrylovKit is provided by the following functions:\n\nlinsolve: solve linear systems\neigsolve: find a few eigenvalues and corresponding eigenvectors\ngeneigsolve: find a few generalized eigenvalues and corresponding vectors\nsvdsolve: find a few singular values and corresponding left and right   singular vectors\nexponentiate: apply the exponential of a linear map to a vector\n\n\n\n\n\n"
},

{
    "location": "man/intro/#Getting-started-1",
    "page": "Introduction",
    "title": "Getting started",
    "category": "section",
    "text": "After installation, start by loading KrylovKitusing KrylovKitThe help entry of the KrylovKit module statesKrylovKit"
},

{
    "location": "man/intro/#KrylovKit.ConvergenceInfo",
    "page": "Introduction",
    "title": "KrylovKit.ConvergenceInfo",
    "category": "type",
    "text": "struct ConvergenceInfo{S,T}\n    converged::Int\n    residual::T\n    normres::S\n    numiter::Int\n    numops::Int\nend\n\nUsed to return information about the solution found by the iterative method.\n\nconverged: the number of solutions that have converged according to an appropriate error   measure and requested tolerance for the problem. Its value can be zero or one for   linsolve and exponentiate, or any integer >= 0 for   eigsolve, schursolve or svdsolve.\nresidual: the (list of) residual(s) for the problem, or nothing for problems without   the concept of a residual (i.e. exponentiate). This is a single vector (of the same   type as the type of vectors used in the problem) for linsolve, or a Vector of such   vectors for eigsolve, schursolve or svdsolve.\nnormres: the norm of the residual(s) (in the previous field) or the value of any other   error measure that is appropriate for the problem. This is a Real for linsolve and   exponentiate, and a Vector{<:Real} for eigsolve, schursolve and svdsolve. The   number of values in normres that are smaller than a predefined tolerance corresponds   to the number converged of solutions that have converged.\nnumiter: the number of iterations (sometimes called restarts) used by the algorithm.\nnumops: the number of times the linear map or operator was applied\n\n\n\n\n\n"
},

{
    "location": "man/intro/#Common-interface-1",
    "page": "Introduction",
    "title": "Common interface",
    "category": "section",
    "text": "The for high-level function linsolve, eigsolve, geneigsolve, svdsolve and exponentiate follow a common interfaceresults..., info = problemsolver(A, args...; kwargs...)where problemsolver is one of the functions above. Here, A is the linear map in the problem, which could be an instance of AbstractMatrix, or any function or callable object that encodes the action of the linear map on a vector. In particular, one can write the linear map using Julia\'s do block syntax asresults..., info = problemsolver(args...; kwargs...) do x\n    y = # implement linear map on x\n    return y\nendRead the documentation for problems that require both the linear map and its adjoint to be implemented, e.g. svdsolve, or that require two different linear maps, e.g. geneigsolve.Furthermore, args is a set of additional arguments to specify the problem. The keyword arguments kwargs contain information about the linear map (issymmetric, ishermitian, isposdef) and about the solution strategy (tol, krylovdim, maxiter). Finally, there is a keyword argument verbosity that determines how much information is printed to STDOUT. The default value verbosity = 0 means that no information will be printed. With verbosity = 1, a single message at the end of the algorithm will be displayed, which is a warning if the algorithm did not succeed in finding the solution, or some information if it did. For verbosity = 2, information about the current state is displayed after every iteration of the algorithm. Finally, for verbosity > 2, information about the individual Krylov expansion steps is displayed.The return value contains one or more entries that define the solution, and a final entry info of type ConvergeInfo that encodes information about the solution, i.e. wether it has converged, the residual(s) and the norm thereof, the number of operations used:KrylovKit.ConvergenceInfoThere is also an expert interface where the user specifies the algorithm that should be used explicitly, i.e.results..., info = problemsolver(A, args..., algorithm(;kwargs...))Most algorithm constructions take the same keyword arguments (tol, krylovdim, maxiter and verbosity) discussed above."
},

{
    "location": "man/linear/#",
    "page": "Linear problems",
    "title": "Linear problems",
    "category": "page",
    "text": ""
},

{
    "location": "man/linear/#KrylovKit.linsolve",
    "page": "Linear problems",
    "title": "KrylovKit.linsolve",
    "category": "function",
    "text": "linsolve(A::AbstractMatrix, b::AbstractVector, [a₀::Number = 0, a₁::Number = 1,\n            T::Type = promote_type(eltype(A), eltype(b), typeof(a₀), typeof(a₁))];\n            kwargs...)\nlinsolve(f, b, [a₀::Number = 0, a₁::Number = 1,\n            T::Type = promote_type(eltype(b), typeof(a₀), typeof(a₁))]; kwargs...)\nlinsolve(f, b, x₀, [a₀::Number = 0, a₁::Number = 1]; kwargs...)\nlinsolve(f, b, x₀, algorithm, [a₀::Number = 0, a₁::Number = 1])\n\nCompute a solution x to the linear system (a₀ + a₁ * A)*x = b or a₀ * x + a₁ * f(x) = b, possibly using a starting guess x₀. Return the approximate solution x and a ConvergenceInfo structure.\n\nArguments:\n\nThe linear map can be an AbstractMatrix (dense or sparse) or a general function or callable object. If no initial guess is specified, it is chosen as rmul!(similar(b, T), false) which generates a similar object to b, but with element type T and initialized with zeros. The numbers a₀ and a₁ are optional arguments; they are applied implicitly, i.e. they do not contribute the computation time of applying the linear map or to the number of operations on vectors of type x and b.\n\nFinally, the optional argument T acts as a hint in which Number type the computation should be performed, but is not restrictive. If the linear map automatically produces complex values, complex arithmetic will be used even though T<:Real was specified.\n\nReturn values:\n\nThe return value is always of the form x, info = linsolve(...) with\n\nx: the approximate solution to the problem, similar type as the right hand side b   but possibly with a different eltype\ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: takes value 0 or 1 depending on whether the solution was   converged up to the requested tolerance\ninfo.residual: residual b - f(x) of the approximate solution x\ninfo.normres::Real: norm of the residual, i.e. norm(info.residual)\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nNo warning is printed if not all requested eigenvalues were converged, so always check if info.converged == 1.\n\nKeyword arguments:\n\nKeyword arguments are given by:\n\nverbosity::Int = 0: verbosity level, i.e. 0 (no messages), 1 (single message   at the end), 2 (information after every iteration), 3 (information per Krylov step)\natol::Real: the requested accuracy, i.e. absolute tolerance, on the norm of the   residual.\nrtol::Real: the requested accuracy on the norm of the residual, relative to the norm   of the right hand side b.\ntol::Real: the requested accuracy on the norm of the residual which is actually used,   but which defaults to max(atol, rtol*norm(b)). So either atol and rtol or directly   use tol (in which case the value of atol and rtol will be ignored).\nkrylovdim::Integer: the maximum dimension of the Krylov subspace that will be   constructed.\n`maxiter::Integer: the number of times the Krylov subspace can be rebuilt; see below for   further details on the algorithms.\north::Orthogonalizer: the orthogonalization method to be used, see   Orthogonalizer\nissymmetric::Bool: if the linear map is symmetric, only meaningful if T<:Real\nishermitian::Bool: if the linear map is hermitian\nisposdef::Bool: if the linear map is positive definite\n\nThe default values are given by atol = KrylovDefaults.tol, rtol = KrylovDefaults.tol, tol = max(atol, rtol*norm(b)), krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter, orth = KrylovDefaults.orth; see KrylovDefaults for details.\n\nThe default value for the last three parameters depends on the method. If an AbstractMatrix is used, issymmetric, ishermitian and isposdef are checked for that matrix, ortherwise the default values are issymmetric = false, ishermitian = T <: Real && issymmetric and isposdef = false.\n\nAlgorithms\n\nThe last method, without default values and keyword arguments, is the one that is finally called, and can also be used directly. Here, one specifies the algorithm explicitly. Currently, only CG and GMRES are implemented, where CG is chosen if isposdef == true. Note that in standard GMRES terminology, our parameter krylovdim is referred to as the restart parameter, and our maxiter parameter counts the number of outer iterations, i.e. restart cycles. In CG, the Krylov subspace is only implicit because short recurrence relations are being used, and therefore no restarts are required. Therefore, we pass krylovdim*maxiter as the maximal number of CG iterations that can be used by the CG algorithm.\n\n\n\n\n\n"
},

{
    "location": "man/linear/#Linear-problems-1",
    "page": "Linear problems",
    "title": "Linear problems",
    "category": "section",
    "text": "Linear systems are of the form A*x=b where A should be a linear map that has the same type of output as input, i.e. the solution x should be of the same type as the right hand side b. They can be solved using the function linsolve:linsolveCurrently supported algorithms are CG and GMRES."
},

{
    "location": "man/eig/#",
    "page": "Eigenvalue problems",
    "title": "Eigenvalue problems",
    "category": "page",
    "text": ""
},

{
    "location": "man/eig/#Eigenvalue-problems-1",
    "page": "Eigenvalue problems",
    "title": "Eigenvalue problems",
    "category": "section",
    "text": ""
},

{
    "location": "man/eig/#KrylovKit.eigsolve",
    "page": "Eigenvalue problems",
    "title": "KrylovKit.eigsolve",
    "category": "function",
    "text": "eigsolve(A::AbstractMatrix, [howmany = 1, which = :LM, T = eltype(A)]; kwargs...)\neigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)\neigsolve(f, x₀, [howmany = 1, which = :LM]; kwargs...)\neigsolve(f, x₀, howmany, which, algorithm)\n\nCompute at least howmany eigenvalues from the linear map encoded in the matrix A or by the function f. Return eigenvalues, eigenvectors and a ConvergenceInfo structure.\n\nArguments:\n\nThe linear map can be an AbstractMatrix (dense or sparse) or a general function or callable object. If an AbstractMatrix is used, a starting vector x₀ does not need to be provided, it is then chosen as rand(T, size(A,1)). If the linear map is encoded more generally as a a callable function or method, the best approach is to provide an explicit starting guess x₀. Note that x₀ does not need to be of type AbstractVector, any type that behaves as a vector and supports the required methods (see KrylovKit docs) is accepted. If instead of x₀ an integer n is specified, it is assumed that x₀ is a regular vector and it is initialized to rand(T,n), where the default value of T is Float64, unless specified differently.\n\nThe next arguments are optional, but should typically be specified. howmany specifies how many eigenvalues should be computed; which specifies which eigenvalues should be targetted. Valid specifications of which are given by\n\n:LM: eigenvalues of largest magnitude\n:LR: eigenvalues with largest (most positive) real part\n:SR: eigenvalues with smallest (most negative) real part\n:LI: eigenvalues with largest (most positive) imaginary part, only if T <: Complex\n:SI: eigenvalues with smallest (most negative) imaginary part, only if T <: Complex\nEigSorter(f; rev = false): eigenvalues λ that appear first (or last if   rev == true) when sorted by f(λ)\n\nnote: Note about selecting `which` eigenvalues\nKrylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of the spectrum of the linear map. All of they valid Symbols for which have this property, but could also be specified usign EigSorter, e.g. :LM is equivalent to Eigsorter(abs; rev = true). Note that smallest magnitude sorting is obtained using e.g. EigSorter(abs; rev = false), but since no (shift-and)-invert is used, this will only be successfull if you somehow know that eigenvalues close to zero are also close to the periphery of the spectrum.\n\nThe argument T acts as a hint in which Number type the computation should be performed, but is not restrictive. If the linear map automatically produces complex values, complex arithmetic will be used even though T<:Real was specified. However, if the linear map and initial guess are real, approximate eigenvalues will be searched for using a partial Schur factorization, which implies that complex conjugate eigenvalues come in pairs and cannot be split. It is then illegal to choose which in a way that would treat λ and conj(λ) differently, i.e. :LI and :SI are invalid, as well as any EigSorter that would lead to by(λ) != by(conj(λ)).\n\nReturn values:\n\nThe return value is always of the form vals, vecs, info = eigsolve(...) with\n\nvals: a Vector containing the eigenvalues, of length at least howmany, but could   be longer if more eigenvalues were converged at the same cost. Eigenvalues will be real   if Lanczos was used and complex if Arnoldi was used (see below).\nvecs: a Vector of corresponding eigenvectors, of the same length as vals. Note   that eigenvectors are not returned as a matrix, as the linear map could act on any   custom Julia type with vector like behavior, i.e. the elements of the list vecs are   objects that are typically similar to the starting guess x₀, up to a possibly   different eltype. In particular  for a general matrix (i.e. with Arnoldi) the   eigenvectors are generally complex and are therefore always returned in a complex   number format. When the linear map is a simple AbstractMatrix, vecs will be   Vector{Vector{<:Number}}.\ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: indicates how many eigenvalues and eigenvectors were actually   converged to the specified tolerance tol (see below under keyword arguments)\ninfo.residual::Vector: a list of the same length as vals containing the   residuals info.residual[i] = f(vecs[i]) - vals[i] * vecs[i]\ninfo.normres::Vector{<:Real}: list of the same length as vals containing the   norm of the residual info.normres[i] = norm(info.residual[i])\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nNo warning is printed if not all requested eigenvalues were converged, so always check if info.converged >= howmany.\n\nKeyword arguments:\n\nKeyword arguments and their default values are given by:\n\nverbosity::Int = 0: verbosity level, i.e. 0 (no messages), 1 (single message   at the end), 2 (information after every iteration), 3 (information per Krylov step)\ntol::Real: the requested accuracy (corresponding to the 2-norm of the residual for   Schur vectors, not the eigenvectors). If you work in e.g. single precision (Float32),   you should definitely change the default value.\nkrylovdim::Integer: the maximum dimension of the Krylov subspace that will be   constructed. Note that the dimension of the vector space is not known or checked, e.g.   x₀ should not necessarily support the Base.length function. If you know the actual   problem dimension is smaller than the default value, it is useful to reduce the value of   krylovdim, though in principle this should be detected.\nmaxiter::Integer: the number of times the Krylov subspace can be rebuilt; see below   for further details on the algorithms.\north::Orthogonalizer: the orthogonalization method to be used, see   Orthogonalizer\nissymmetric::Bool: if the linear map is symmetric, only meaningful if T<:Real\nishermitian::Bool: if the linear map is hermitian\n\nThe default values are given by tol = KrylovDefaults.tol, krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter, orth = KrylovDefaults.orth; see KrylovDefaults for details.\n\nThe default value for the last two parameters depends on the method. If an AbstractMatrix is used, issymmetric and ishermitian are checked for that matrix, ortherwise the default values are issymmetric = false and ishermitian = T <: Real && issymmetric. When values for the keyword arguments are provided, no checks will be performed even in the matrix case.\n\nAlgorithm\n\nThe last method, without default values and keyword arguments, is the one that is finally called, and can also be used directly. Here, one specifies the algorithm explicitly as either Lanczos, for real symmetric or complex hermitian problems, or Arnoldi, for general problems. Note that these names refer to the process for building the Krylov subspace, but the actual algorithm is an implementation of the Krylov-Schur algorithm, which can dynamically shrink and grow the Krylov subspace, i.e. the restarts are so-called thick restarts where a part of the current Krylov subspace is kept.\n\nnote: Note about convergence\nIn case of a general problem, where the Arnoldi method is used, convergence of an eigenvalue is not based on the norm of the residual norm(f(vecs[i]) - vals[i]*vecs[i]) for the eigenvector but rather on the norm of the residual for the corresponding Schur vectors.See also schursolve if you want to use the partial Schur decomposition directly, or if you are not interested in computing the eigenvectors, and want to work in real arithmetic all the way true (if the linear map and starting guess are real).\n\n\n\n\n\n"
},

{
    "location": "man/eig/#KrylovKit.EigSorter",
    "page": "Eigenvalue problems",
    "title": "KrylovKit.EigSorter",
    "category": "type",
    "text": "EigSorter(by; rev = false)\n\nA simple struct to be used in combination with eigsolve or schursolve to indicate which eigenvalues need to be targetted, namely those that appear first when sorted by by and possibly in reverse order if rev == true.\n\n\n\n\n\n"
},

{
    "location": "man/eig/#KrylovKit.schursolve",
    "page": "Eigenvalue problems",
    "title": "KrylovKit.schursolve",
    "category": "function",
    "text": "schursolve(A, x₀, howmany, which, algorithm)\n\nCompute a partial Schur decomposition containing howmany eigenvalues from the linear map encoded in the matrix or function A. Return the reduced Schur matrix, the basis of Schur vectors, the extracted eigenvalues and a ConvergenceInfo structure.\n\nSee also eigsolve to obtain the eigenvectors instead. For real symmetric or complex hermitian problems, the (partial) Schur decomposition is identical to the (partial) eigenvalue decomposition, and eigsolve should always be used.\n\nArguments:\n\nThe linear map can be an AbstractMatrix (dense or sparse) or a general function or callable object, that acts on vector like objects similar to x₀, which is the starting guess from which a Krylov subspace will be built. howmany specifies how many Schur vectors should be converged before the algorithm terminates; which specifies which eigenvalues should be targetted. Valid specifications of which are\n\nLM: eigenvalues of largest magnitude\nLR: eigenvalues with largest (most positive) real part\nSR: eigenvalues with smallest (most negative) real part\nLI: eigenvalues with largest (most positive) imaginary part, only if T <: Complex\nSI: eigenvalues with smallest (most negative) imaginary part, only if T <: Complex\nClosestTo(λ): eigenvalues closest to some number λ\n\nnote: Note about selecting `which` eigenvalues\nKrylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of the spectrum of the linear map. Even with ClosestTo, no shift and invert is performed. This is useful if, e.g., you know the spectrum to be within the unit circle in the complex plane, and want to target the eigenvalues closest to the value λ = 1.\n\nThe final argument algorithm can currently only be an instance of Arnoldi, but should nevertheless be specified. Since schursolve is less commonly used as eigsolve, no convenient keyword syntax is currently available.\n\nReturn values:\n\nThe return value is always of the form T, vecs, vals, info = schursolve(...) with\n\nT: a Matrix containing the partial Schur decomposition of the linear map, i.e. it\'s   elements are given by T[i,j] = dot(vecs[i], f(vecs[j])). It is of Schur form, i.e.   upper triangular in case of complex arithmetic, and block upper triangular (with at most   2x2 blocks) in case of real arithmetic.\nvecs: a Vector of corresponding Schur vectors, of the same length as vals. Note   that Schur vectors are not returned as a matrix, as the linear map could act on any   custom  Julia type with vector like behavior, i.e. the elements of the list vecs are   objects that are typically similar to the starting guess x₀, up to a possibly   different eltype. When the linear map is a simple AbstractMatrix, vecs will be   Vector{Vector{<:Number}}. Schur vectors are by definition orthogonal, i.e.   dot(vecs[i],vecs[j]) = I[i,j]. Note that Schur vectors are real if the problem (i.e.   the linear map and the initial guess) are real.\nvals: a Vector of eigenvalues, i.e. the diagonal elements of T in case of complex   arithmetic, or extracted from the diagonal blocks in case of real arithmetic. Note that   vals will always be complex, independent of the underlying arithmetic.\ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: indicates how many eigenvalues and Schur vectors were   actually converged to the specified tolerance (see below under keyword arguments)\ninfo.residuals::Vector: a list of the same length as vals containing the actual   residuals   julia     info.residuals[i] = f(vecs[i]) - sum(vecs[j]*T[j,i] for j = 1:i+1)   where T[i+1,i] is definitely zero in case of complex arithmetic and possibly zero   in case of real arithmetic\ninfo.normres::Vector{<:Real}: list of the same length as vals containing the   norm of the residual for every Schur vector, i.e.   info.normes[i] = norm(info.residual[i])\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nNo warning is printed if not all requested eigenvalues were converged, so always check if info.converged >= howmany.\n\nAlgorithm\n\nThe actual algorithm is an implementation of the Krylov-Schur algorithm, where the Arnoldi algorithm is used to generate the Krylov subspace. During the algorith, the Krylov subspace is dynamically grown and shrunk, i.e. the restarts are so-called thick restarts where a part of the current Krylov subspace is kept.\n\n\n\n\n\n"
},

{
    "location": "man/eig/#Eigenvalues-and-eigenvectors-1",
    "page": "Eigenvalue problems",
    "title": "Eigenvalues and eigenvectors",
    "category": "section",
    "text": "Finding a selection of eigenvalues and corresponding (right) eigenvectors of a linear map can be accomplished with the eigsolve routine:eigsolveWhich eigenvalues are targetted can be specified using one of the symbols :LM, :LR, :SR, :LI and :SI for largest magnitude, largest and smallest real part, and largest and smallest imaginary part respectively. Alternatively, one can just specify a general sorting operation using EigSorterEigSorterFor a general matrix, eigenvalues and eigenvectors will always be returned with complex values for reasons of type stability. However, if the linear map and initial guess are real, most of the computation is actually performed using real arithmetic, as in fact the first step is to compute an approximate partial Schur factorization. If one is not interested in the eigenvectors, one can also just compute this partial Schur factorization using schursolve.schursolveNote that, for symmetric or hermitian linear maps, the eigenvalue and Schur factorization are equivalent, and one can only use eigsolve.Another example of a possible use case of schursolve is if the linear map is known to have a unique eigenvalue of, e.g. largest magnitude. Then, if the linear map is real valued, that largest magnitude eigenvalue and its corresponding eigenvector are also real valued. eigsolve will automatically return complex valued eigenvectors for reasons of type stability. However, as the first Schur vector will coincide with the first eigenvector, one can instead useT, vecs, vals, info = schursolve(A, x⁠₀, 1, :LM, Arnoldi(...))and use vecs[1] as the real valued eigenvector (after checking info.converged) corresponding to the largest magnitude eigenvalue of A."
},

{
    "location": "man/eig/#KrylovKit.geneigsolve",
    "page": "Eigenvalue problems",
    "title": "KrylovKit.geneigsolve",
    "category": "function",
    "text": "geneigsolve((A::AbstractMatrix, B::AbstractMatrix), [howmany = 1, which = :LM,\n                                T = promote_type(eltype(A), eltype(B))]; kwargs...)\ngeneigsolve(f, n::Int, [howmany = 1, which = :LM, T = Float64]; kwargs...)\ngeneigsolve(f, x₀, [howmany = 1, which = :LM]; kwargs...)\ngeneigsolve(f, x₀, howmany, which, algorithm)\n\nCompute at least howmany generalized eigenvalues λ and generalized eigenvectors x of the form (A - λB)x = 0, where A and B are either instances of AbstractMatrix, or some function that implements the matrix vector product. In case functions are used, one could either specify the action of A and B via a tuple of two functions (or a function and an AbstractMatrix), or one could use a single function that takes a single argument x and returns two results, corresponding to A*x and B*x. Return the computed eigenvalues, eigenvectors and a ConvergenceInfo structure.\n\nArguments:\n\nThe first argument is either a tuple of two linear maps, so a function or an AbstractMatrix for either of them, representing the action of A and B. Alternatively, a single function can be used that takes a single argument x and returns the equivalent of (A*x, B*x) as result. This latter form is compatible with the do block syntax of Julia. If an AbstractMatrix is used for either A or B, a starting vector x₀ does not need to be provided, it is then chosen as rand(T, size(A,1)) if A is an AbstractMatrix (or similarly if only B is an AbstractMatrix). Here T = promote_type(eltype(A), eltype(B)) if both A and B are instances of AbstractMatrix, or just the eltype of whichever is an AbstractMatrix. If both A and B are encoded more generally as a callable function or method, the bestapproach is to provide an explicit starting guess x₀. Note that x₀ does not need to be of type AbstractVector, any type that behaves as a vector and supports the required methods (see KrylovKit docs) is accepted. If instead of x₀ an integer n is specified, it is assumed that x₀ is a regular vector and it is initialized to rand(T,n), where the default value of T is Float64, unless specified differently.\n\nThe next arguments are optional, but should typically be specified. howmany specifies how many eigenvalues should be computed; which specifies which eigenvalues should be targetted. Valid specifications of which are given by\n\n:LM: eigenvalues of largest magnitude\n:LR: eigenvalues with largest (most positive) real part\n:SR: eigenvalues with smallest (most negative) real part\n:LI: eigenvalues with largest (most positive) imaginary part, only if T <: Complex\n:SI: eigenvalues with smallest (most negative) imaginary part, only if T <: Complex\nEigSorter(f; rev = false): eigenvalues λ that appear first (or last if   rev == true) when sorted by f(λ)\n\nnote: Note about selecting `which` eigenvalues\nKrylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of the spectrum of the linear map. Even with ClosestTo, no shift and invert is performed. This is useful if, e.g., you know the spectrum to be within the unit circle in the complex plane, and want to target the eigenvalues closest to the value λ = 1.\n\nThe argument T acts as a hint in which Number type the computation should be performed, but is not restrictive. If the linear map automatically produces complex values, complex arithmetic will be used even though T<:Real was specified.\n\nReturn values:\n\nThe return value is always of the form vals, vecs, info = geneigsolve(...) with\n\nvals: a Vector containing the eigenvalues, of length at least howmany, but could   be longer if more eigenvalues were converged at the same cost.\nvecs: a Vector of corresponding eigenvectors, of the same length as vals.   Note that eigenvectors are not returned as a matrix, as the linear map could act on any   custom Julia type with vector like behavior, i.e. the elements of the list vecs are   objects that are typically similar to the starting guess x₀, up to a possibly   different eltype. When the linear map is a simple AbstractMatrix, vecs will be   Vector{Vector{<:Number}}.\ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: indicates how many eigenvalues and eigenvectors were actually   converged to the specified tolerance tol (see below under keyword arguments)\ninfo.residual::Vector: a list of the same length as vals containing the   residuals info.residual[i] = f(vecs[i]) - vals[i] * vecs[i]\ninfo.normres::Vector{<:Real}: list of the same length as vals containing the   norm of the residual info.normres[i] = norm(info.residual[i])\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nNo warning is printed if not all requested eigenvalues were converged, so always check if info.converged >= howmany.\n\nKeyword arguments:\n\nKeyword arguments and their default values are given by:\n\nverbosity::Int = 0: verbosity level, i.e. 0 (no messages), 1 (single message   at the end), 2 (information after every iteration), 3 (information per Krylov step)\ntol::Real: the requested accuracy (corresponding to the 2-norm of the residual for   Schur vectors, not the eigenvectors). If you work in e.g. single precision (Float32),   you should definitely change the default value.\nkrylovdim::Integer: the maximum dimension of the Krylov subspace that will be   constructed. Note that the dimension of the vector space is not known or checked, e.g.   x₀ should not necessarily support the Base.length function. If you know the actual   problem dimension is smaller than the default value, it is useful to reduce the value   of krylovdim, though in principle this should be detected.\nmaxiter::Integer: the number of times the Krylov subspace can be rebuilt; see below   for further details on the algorithms.\north::Orthogonalizer: the orthogonalization method to be used, see   Orthogonalizer\nissymmetric::Bool: if both linear maps A and B are symmetric, only meaningful if   T<:Real\nishermitian::Bool: if both linear maps A and B are hermitian\nisposdef::Bool: if the linear map B is positive definite\n\nThe default values are given by tol = KrylovDefaults.tol, krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter, orth = KrylovDefaults.orth; see KrylovDefaults for details.\n\nThe default value for the last three parameters depends on the method. If an AbstractMatrix is used, issymmetric, ishermitian and isposdef are checked for that matrix, ortherwise the default values are issymmetric = false and ishermitian = T <: Real && issymmetric. When values are provided, no checks will be performed even in the matrix case.\n\nAlgorithm\n\nThe last method, without default values and keyword arguments, is the one that is finally called, and can also be used directly. Here the algorithm is specified, though currently only GolubYe is available. The Golub-Ye algorithm is an algorithm for solving hermitian (symmetric) generalized eigenvalue problems A x = λ B x with positive definite B, without the need for inverting B. It builds a Krylov subspace of size krylovdim starting from an estimate x by acting with (A - ρ(x) B), where ρ(x) = dot(x, A*x)/ dot(x, B*x), and employing the Lanczos algorithm. This process is repeated at most maxiter times. In every iteration k>1, the subspace will also be expanded to size krylovdim+1 by adding x_k - x_k-1, which is known as the LOPCG correction and was suggested by Money and Ye. With krylovdim = 2, this algorithm becomes equivalent to LOPCG.\n\nwarning: Restriction to symmetric definite generalized eigenvalue problems\nWhile the only algorithm so far is restricted to symmetric/hermitian generalized eigenvalue problems with positive definite B, this is not reflected in the default values for the keyword arguments issymmetric or ishermitian and isposdef. Make sure to set these to true to understand the implications of using this algorithm.\n\n\n\n\n\n"
},

{
    "location": "man/eig/#Generalized-eigenvalue-problems-1",
    "page": "Eigenvalue problems",
    "title": "Generalized eigenvalue problems",
    "category": "section",
    "text": "Generalized eigenvalues λ and corresponding vectors x of the generalized eigenvalue problem A x = λ B x can be obtained using the method geneigsolve. Currently, there is only one algorithm, which does not require inverses of A or B, but is restricted to symmetric or hermitian generalized eigenvalue problems where the matrix or linear map B is positive definite. Note that this is not reflected in the default values for the keyword arguments issymmetric, ishermitian and isposdef, so that these should be set explicitly in order to comply with this restriction. If A and B are actual instances of AbstractMatrix, the default value for the keyword arguments will try to check these properties explicitly.geneigsolve"
},

{
    "location": "man/svd/#",
    "page": "Singular value problems",
    "title": "Singular value problems",
    "category": "page",
    "text": ""
},

{
    "location": "man/svd/#KrylovKit.svdsolve",
    "page": "Singular value problems",
    "title": "KrylovKit.svdsolve",
    "category": "function",
    "text": "svdsolve(A::AbstractMatrix, [howmany = 1, which = :LR, T = eltype(A)]; kwargs...)\nsvdsolve(f, m::Int, n::Int, [howmany = 1, which = :LR, T = Float64]; kwargs...)\nsvdsolve(f, x₀, y₀, [howmany = 1, which = :LM]; kwargs...)\nsvdsolve(f, x₀, y₀, howmany, which, algorithm)\n\nCompute howmany singular values from the linear map encoded in the matrix A or by the function f. Return singular values, left and right singular vectors and a ConvergenceInfo structure.\n\nArguments:\n\nThe linear map can be an AbstractMatrix (dense or sparse) or a general function or callable object. Since both the action of the linear map and its adjoint are required in order to compute singular values, f can either be a tuple of two callable objects (each accepting a single argument), representing the linear map and its adjoint respectively, or, f can be a single callable object that accepts two input arguments, where the second argument is a flag that indicates whether the adjoint or the normal action of the linear map needs to be computed. The latter form still combines well with the do block syntax of Julia, as in\n\nvals, lvecs, rvecs, info = svdsolve(x₀, y₀, howmany, which; kwargs...) do (x, flag)\n    if flag\n        # y = compute action of adjoint map on x\n    else\n        # y = compute action of linear map on x\n    end\n    return y\nend\n\nFor a general linear map encoded using either the tuple or the two-argument form, the best approach is to provide a start vector x₀ (in the domain of the linear map). Alternatively, one can specify the number n of columns of the linear map, in which case x₀ = rand(T, n) is used, where the default value of T is Float64, unless specified differently. If an AbstractMatrix is used, a starting vector x₀ does not need to be provided; it is chosen as rand(T, size(A,1)).\n\nThe next arguments are optional, but should typically be specified. howmany specifies how many singular values and vectors should be computed; which specifies which singular values should be targetted. Valid specifications of which are\n\nLR: largest singular values\nSR: smallest singular values\n\nHowever, the largest singular values tend to converge more rapidly.\n\nReturn values:\n\nThe return value is always of the form vals, lvecs, rvecs, info = svdsolve(...) with\n\nvals: a Vector{<:Real} containing the singular values, of length at least howmany,   but could be longer if more singular values were converged at the same cost.\nlvecs: a Vector of corresponding left singular vectors, of the same length as   vals.\nrvecs: a Vector of corresponding right singular vectors, of the same length as   vals. Note that singular vectors are not returned as a matrix, as the linear map   could act on any custom Julia type with vector like behavior, i.e. the elements of the   lists lvecs(rvecs) are objects that are typically similar to the starting guess y₀   (x₀), up to a possibly different eltype. When the linear map is a simple   AbstractMatrix, lvecs and rvecs will be Vector{Vector{<:Number}}.\ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: indicates how many singular values and vectors were actually   converged to the specified tolerance tol (see below under keyword arguments)\ninfo.residual::Vector: a list of the same length as vals containing the   residuals   info.residual[i] = A * rvecs[i] - vals[i] * lvecs[i].\ninfo.normres::Vector{<:Real}: list of the same length as vals containing the   norm of the residual info.normres[i] = norm(info.residual[i])\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A or A\'.\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nNo warning is printed if not all requested eigenvalues were converged, so always check if info.converged >= howmany.\n\nKeyword arguments:\n\nKeyword arguments and their default values are given by:\n\nverbosity::Int = 0: verbosity level, i.e. 0 (no messages), 1 (single message   at the end), 2 (information after every iteration), 3 (information per Krylov step)\nkrylovdim: the maximum dimension of the Krylov subspace that will be constructed.   Note that the dimension of the vector space is not known or checked, e.g. x₀ should not   necessarily support the Base.length function. If you know the actual problem dimension   is smaller than the default value, it is useful to reduce the value of krylovdim, though   in principle this should be detected.\ntol: the requested accuracy according to normres as defined above. If you work in   e.g. single precision (Float32), you should definitely change the default value.\nmaxiter: the number of times the Krylov subspace can be rebuilt; see below for further   details on the algorithms.\north: the orthogonalization method to be used, see Orthogonalizer\n\nAlgorithm\n\nThe last method, without default values and keyword arguments, is the one that is finally called, and can also be used directly. Here the algorithm is specified, though currently only GKL is available. GKL refers to the the partial Golub-Kahan-Lanczos bidiagonalization which forms the basis for computing the approximation to the singular values. This factorization is dynamically shrunk and expanded (i.e. thick restart) similar to the Krylov-Schur factorization for eigenvalues.\n\n\n\n\n\n"
},

{
    "location": "man/svd/#Singular-value-problems-1",
    "page": "Singular value problems",
    "title": "Singular value problems",
    "category": "section",
    "text": "Computing a few singular values and corresponding left and right singular vectors is done using the function svdsolve:svdsolve"
},

{
    "location": "man/matfun/#",
    "page": "Functions of matrices and linear maps",
    "title": "Functions of matrices and linear maps",
    "category": "page",
    "text": ""
},

{
    "location": "man/matfun/#KrylovKit.exponentiate",
    "page": "Functions of matrices and linear maps",
    "title": "KrylovKit.exponentiate",
    "category": "function",
    "text": "function exponentiate(A, t::Number, x; kwargs...)\nfunction exponentiate(A, t::Number, x, algorithm)\n\nCompute y = exp(t*A) x, where A is a general linear map, i.e. a AbstractMatrix or just a general function or callable object and x is of any Julia type with vector like behavior.\n\nArguments:\n\nThe linear map A can be an AbstractMatrix (dense or sparse) or a general function or callable object that implements the action of the linear map on a vector. If A is an AbstractMatrix, x is expected to be an AbstractVector, otherwise x can be of any type that behaves as a vector and supports the required methods (see KrylovKit docs).\n\nThe time parameter t can be real or complex, and it is better to choose t e.g. imaginary and A hermitian, then to absorb the imaginary unit in an antihermitian A. For the former, the Lanczos scheme is used to built a Krylov subspace, in which an approximation to the exponential action of the linear map is obtained. The argument x can be of any type and should be in the domain of A.\n\nReturn values:\n\nThe return value is always of the form y, info = exponentiate(...) with\n\ny: the result of the computation, i.e. y = exp(t*A)*x\ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: 0 or 1 if the solution y was approximated up to the   requested tolerance tol.\ninfo.residual::Nothing: value nothing, there is no concept of a residual in   this case\ninfo.normres::Real: a (rough) estimate of the error between the approximate and   exact solution\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nBy default (i.e. if verbosity = 0, see below), no warning is printed if the solution was not found with the requested precion, so be sure to check info.converged == 1.\n\nKeyword arguments:\n\nKeyword arguments and their default values are given by:\n\nverbosity::Int = 0: verbosity level, i.e. 0 (no messages), 1 (single message   at the end), 2 (information after every iteration), 3 (information per Krylov step)\nkrylovdim = 30: the maximum dimension of the Krylov subspace that will be constructed.   Note that the dimension of the vector space is not known or checked, e.g. x₀ should   not necessarily support the Base.length function. If you know the actual problem   dimension is smaller than the default value, it is useful to reduce the value of   krylovdim, though in principle this should be detected.\ntol = 1e-12: the requested accuracy per unit time, i.e. if you want a certain   precision ϵ on the final result, set tol = ϵ/abs(t). If you work in e.g. single   precision (Float32), you should definitely change the default value.\nmaxiter::Int = 100: the number of times the Krylov subspace can be rebuilt; see below   for further details on the algorithms.\nissymmetric: if the linear map is symmetric, only meaningful if T<:Real\nishermitian: if the linear map is hermitian\n\nThe default value for the last two depends on the method. If an AbstractMatrix is used, issymmetric and ishermitian are checked for that matrix, ortherwise the default values are issymmetric = false and ishermitian = T <: Real && issymmetric.\n\nAlgorithm\n\nThis is actually a simple wrapper over more general method expintegrator for for integrating a linear non-homogeneous ODE.\n\n\n\n\n\n"
},

{
    "location": "man/matfun/#KrylovKit.expintegrator",
    "page": "Functions of matrices and linear maps",
    "title": "KrylovKit.expintegrator",
    "category": "function",
    "text": "function expintegrator(A, t::Number, u₀, u₁, …; kwargs...)\nfunction expintegrator(A, t::Number, (u₀, u₁, …); kwargs...)\nfunction expintegrator(A, t::Number, (u₀, u₁, …), algorithm)\n\nCompute y = ϕ₀(t*A)*u₀ + t*ϕ₁(t*A)*u₁ + t^2*ϕ₂(t*A)*u₂ + , where A is a general linear map, i.e. a AbstractMatrix or just a general function or callable object and u₀, u₁ are of any Julia type with vector like behavior. Here, ϕ₀(z) = exp(z) and ϕⱼ₁ = (ϕⱼ(z) - 1j)z. In particular, y = x(t) represents the solution of the ODE x = A*x + ⱼ t^jj uⱼ₁ with x(0) = u₀.\n\nArguments:\n\nThe linear map A can be an AbstractMatrix (dense or sparse) or a general function or callable object that implements the action of the linear map on a vector. If A is an AbstractMatrix, x is expected to be an AbstractVector, otherwise x can be of any type that behaves as a vector and supports the required methods (see KrylovKit docs).\n\nThe time parameter t can be real or complex, and it is better to choose t e.g. imaginary and A hermitian, then to absorb the imaginary unit in an antihermitian A. For the former, the Lanczos scheme is used to built a Krylov subspace, in which an approximation to the exponential action of the linear map is obtained. The arguments u₀, u₁, … can be of any type and should be in the domain of A.\n\nReturn values:\n\nThe return value is always of the form y, info = expintegrator(...) with\n\ny: the result of the computation, i.e.   y = ϕ₀(t*A)*u₀ + t*ϕ₁(t*A)*u₁ + t^2*ϕ₂(t*A)*u₂ + \ninfo: an object of type [ConvergenceInfo], which has the following fields\ninfo.converged::Int: 0 or 1 if the solution y was approximated up to the   requested tolerance tol.\ninfo.residual::Nothing: value nothing, there is no concept of a residual in   this case\ninfo.normres::Real: a (rough) estimate of the error between the approximate and   exact solution\ninfo.numops::Int: number of times the linear map was applied, i.e. number of times   f was called, or a vector was multiplied with A\ninfo.numiter::Int: number of times the Krylov subspace was restarted (see below)\n\nwarning: Check for convergence\nBy default (i.e. if verbosity = 0, see below), no warning is printed if the solution was not found with the requested precion, so be sure to check info.converged == 1.\n\nKeyword arguments:\n\nKeyword arguments and their default values are given by:\n\nverbosity::Int = 0: verbosity level, i.e. 0 (no messages), 1 (single message   at the end), 2 (information after every iteration), 3 (information per Krylov step)\nkrylovdim = 30: the maximum dimension of the Krylov subspace that will be constructed.   Note that the dimension of the vector space is not known or checked, e.g. x₀ should   not necessarily support the Base.length function. If you know the actual problem   dimension is smaller than the default value, it is useful to reduce the value of   krylovdim, though in principle this should be detected.\ntol = 1e-12: the requested accuracy per unit time, i.e. if you want a certain   precision ϵ on the final result, set tol = ϵ/abs(t). If you work in e.g. single   precision (Float32), you should definitely change the default value.\nmaxiter::Int = 100: the number of times the Krylov subspace can be rebuilt; see below   for further details on the algorithms.\nissymmetric: if the linear map is symmetric, only meaningful if T<:Real\nishermitian: if the linear map is hermitian\n\nThe default value for the last two depends on the method. If an AbstractMatrix is used, issymmetric and ishermitian are checked for that matrix, ortherwise the default values are issymmetric = false and ishermitian = T <: Real && issymmetric.\n\nAlgorithm\n\nThe last method, without keyword arguments and the different vectors u₀, u₁, … in a tuple, is the one that is finally called, and can also be used directly. Here, one specifies the algorithm explicitly as either Lanczos, for real symmetric or complex hermitian linear maps, or Arnoldi, for general linear maps. Note that these names refer to the process for building the Krylov subspace, and that one can still use complex time steps in combination with e.g. a real symmetric map.\n\n\n\n\n\n"
},

{
    "location": "man/matfun/#Functions-of-matrices-and-linear-maps-1",
    "page": "Functions of matrices and linear maps",
    "title": "Functions of matrices and linear maps",
    "category": "section",
    "text": "Currently, the only family of functions of a linear map for which a method is available are the so-called ϕⱼ(z) functions which generalize the exponential function ϕ₀(z) = exp(z). These functions arise in the context of linear non-homogeneous ODEs, and the corresponding Krylov method is an exponentional integrator, hence expintegrator. for a linear homogeneous ODE, the solution is a pure exponential, and the special wrapper exponentiate is available:exponentiate\nexpintegrator"
},

{
    "location": "man/algorithms/#",
    "page": "Available algorithms",
    "title": "Available algorithms",
    "category": "page",
    "text": ""
},

{
    "location": "man/algorithms/#Available-algorithms-1",
    "page": "Available algorithms",
    "title": "Available algorithms",
    "category": "section",
    "text": ""
},

{
    "location": "man/algorithms/#KrylovKit.ClassicalGramSchmidt",
    "page": "Available algorithms",
    "title": "KrylovKit.ClassicalGramSchmidt",
    "category": "type",
    "text": "ClassicalGramSchmidt()\n\nRepresents the classical Gram Schmidt algorithm for orthogonalizing different vectors, typically not an optimal choice.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.ModifiedGramSchmidt",
    "page": "Available algorithms",
    "title": "KrylovKit.ModifiedGramSchmidt",
    "category": "type",
    "text": "ModifiedGramSchmidt()\n\nRepresents the modified Gram Schmidt algorithm for orthogonalizing different vectors, typically a reasonable choice for linear systems but not for eigenvalue solvers with a large Krylov dimension.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.ClassicalGramSchmidt2",
    "page": "Available algorithms",
    "title": "KrylovKit.ClassicalGramSchmidt2",
    "category": "type",
    "text": "ClassicalGramSchmidt2()\n\nRepresents the classical Gram Schmidt algorithm with a second reorthogonalization step always taking place.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.ModifiedGramSchmidt2",
    "page": "Available algorithms",
    "title": "KrylovKit.ModifiedGramSchmidt2",
    "category": "type",
    "text": "ModifiedGramSchmidt2()\n\nRepresents the modified Gram Schmidt algorithm with a second reorthogonalization step always taking place.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.ClassicalGramSchmidtIR",
    "page": "Available algorithms",
    "title": "KrylovKit.ClassicalGramSchmidtIR",
    "category": "type",
    "text": "ClassicalGramSchmidtIR(η::Real = 1/sqrt(2))\n\nRepresents the classical Gram Schmidt algorithm with iterative (i.e. zero or more) reorthogonalization until the norm of the vector after an orthogonalization step has not decreased by a factor smaller than η with respect to the norm before the step. The default value corresponds to the Daniel-Gragg-Kaufman-Stewart condition.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.ModifiedGramSchmidtIR",
    "page": "Available algorithms",
    "title": "KrylovKit.ModifiedGramSchmidtIR",
    "category": "type",
    "text": "ModifiedGramSchmidtIR(η::Real = 1/sqrt(2))\n\nRepresents the modified Gram Schmidt algorithm with iterative (i.e. zero or more) reorthogonalization until the norm of the vector after an orthogonalization step has not decreased by a factor smaller than η with respect to the norm before the step. The default value corresponds to the Daniel-Gragg-Kaufman-Stewart condition.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#Orthogonalization-algorithms-1",
    "page": "Available algorithms",
    "title": "Orthogonalization algorithms",
    "category": "section",
    "text": "ClassicalGramSchmidt\nModifiedGramSchmidt\nClassicalGramSchmidt2\nModifiedGramSchmidt2\nClassicalGramSchmidtIR\nModifiedGramSchmidtIR"
},

{
    "location": "man/algorithms/#KrylovKit.Lanczos",
    "page": "Available algorithms",
    "title": "KrylovKit.Lanczos",
    "category": "type",
    "text": "Lanczos(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,\n    tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)\n\nRepresents the Lanczos algorithm for building the Krylov subspace; assumes the linear operator is real symmetric or complex Hermitian. Can be used in eigsolve and exponentiate. The corresponding algorithms will build a Krylov subspace of size at most krylovdim, which will be repeated at most maxiter times and will stop when the norm of the residual of the Lanczos factorization is smaller than tol. The orthogonalizer orth will be used to orthogonalize the different Krylov vectors. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nUse Arnoldi for non-symmetric or non-Hermitian linear operators.\n\nSee also: factorize, eigsolve, exponentiate, Arnoldi, Orthogonalizer\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.Arnoldi",
    "page": "Available algorithms",
    "title": "KrylovKit.Arnoldi",
    "category": "type",
    "text": "Arnoldi(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,\n    tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)\n\nRepresents the Arnoldi algorithm for building the Krylov subspace for a general matrix or linear operator. Can be used in eigsolve and exponentiate. The corresponding algorithms will build a Krylov subspace of size at most krylovdim, which will be repeated at most maxiter times and will stop when the norm of the residual of the Arnoldi factorization is smaller than tol. The orthogonalizer orth will be used to orthogonalize the different Krylov vectors. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nUse Lanczos for real symmetric or complex Hermitian linear operators.\n\nSee also: eigsolve, exponentiate, Lanczos, Orthogonalizer\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#General-Krylov-algorithms-1",
    "page": "Available algorithms",
    "title": "General Krylov algorithms",
    "category": "section",
    "text": "Lanczos\nArnoldi"
},

{
    "location": "man/algorithms/#KrylovKit.CG",
    "page": "Available algorithms",
    "title": "KrylovKit.CG",
    "category": "type",
    "text": "CG(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)\n\nConstruct an instance of the conjugate gradient algorithm with specified parameters, which can be passed to linsolve in order to iteratively solve a linear system with a positive definite (and thus symmetric or hermitian) coefficent matrix or operator. The CG method will search for the optimal x in a Krylov subspace of maximal size maxiter, or stop when norm(A*x - b) < tol. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nSee also: linsolve, MINRES, GMRES, BiCG, BiCGStab\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.MINRES",
    "page": "Available algorithms",
    "title": "KrylovKit.MINRES",
    "category": "type",
    "text": "MINRES(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)\n\nConstruct an instance of the conjugate gradient algorithm with specified parameters, which can be passed to linsolve in order to iteratively solve a linear system with a real symmetric or complex hermitian coefficent matrix or operator. The MINRES method will search for the optimal x in a Krylov subspace of maximal size maxiter, or stop when norm(A*x - b) < tol. In building the Krylov subspace, MINRES will use the orthogonalizer orth. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nwarning: Not implemented yet\n\n\nSee also: linsolve, CG, GMRES, BiCG, BiCGStab\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.GMRES",
    "page": "Available algorithms",
    "title": "KrylovKit.GMRES",
    "category": "type",
    "text": "GMRES(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,\n    tol = KrylovDefaults.tol, orth::Orthogonalizer = KrylovDefaults.orth)\n\nConstruct an instance of the GMRES algorithm with specified parameters, which can be passed to linsolve in order to iteratively solve a linear system. The GMRES method will search for the optimal x in a Krylov subspace of maximal size krylovdim, and repeat this process for at most maxiter times, or stop when norm(A*x - b) < tol. In building the Krylov subspace, GMRES will use the orthogonalizer orth. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nNote that in the traditional nomenclature of GMRES, the parameter krylovdim is referred to as the restart parameter, and maxiter is the number of outer iterations, i.e. restart cycles. The total iteration count, i.e. the number of expansion steps, is roughly krylovdim times the number of iterations.\n\nSee also: linsolve, BiCG, BiCGStab, CG, MINRES\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.BiCG",
    "page": "Available algorithms",
    "title": "KrylovKit.BiCG",
    "category": "type",
    "text": "BiCG(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)\n\nConstruct an instance of the Biconjugate gradient algorithm with specified parameters, which can be passed to linsolve in order to iteratively solve a linear system general linear map, of which the adjoint can also be applied. The BiCG method will search for the optimal x in a Krylov subspace of maximal size maxiter, or stop when norm(A*x - b) < tol. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nwarning: Not implemented yet\n\n\nSee also: linsolve, BiCGStab, GMRES, CG, MINRES\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#KrylovKit.BiCGStab",
    "page": "Available algorithms",
    "title": "KrylovKit.BiCGStab",
    "category": "type",
    "text": "BiCGStab(; maxiter = KrylovDefaults.maxiter, tol = KrylovDefaults.tol)\n\nConstruct an instance of the Biconjugate gradient algorithm with specified parameters, which can be passed to linsolve in order to iteratively solve a linear system general linear map. The BiCGStab method will search for the optimal x in a Krylov subspace of maximal size maxiter, or stop when norm(A*x - b) < tol. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nwarning: Not implemented yet\n\n\nSee also: linsolve, BiCG, GMRES, CG, MINRES\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#Specific-algorithms-for-linear-systems-1",
    "page": "Available algorithms",
    "title": "Specific algorithms for linear systems",
    "category": "section",
    "text": "CG\nKrylovKit.MINRES\nGMRES\nKrylovKit.BiCG\nKrylovKit.BiCGStab"
},

{
    "location": "man/algorithms/#KrylovKit.GolubYe",
    "page": "Available algorithms",
    "title": "KrylovKit.GolubYe",
    "category": "type",
    "text": "GolubYe(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,\n    tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)\n\nRepresents the Golub-Ye algorithm for solving hermitian (symmetric) generalized eigenvalue problems A x = λ B x with positive definite B, without the need for inverting B. Builds a Krylov subspace of size krylovdim starting from an estimate x by acting with (A - ρ(x) B), where ρ(x) = dot(x, A*x)/dot(x, B*x), and employing the Lanczos algorithm. This process is repeated at most maxiter times. In every iteration k>1, the subspace will also be expanded to size krylovdim+1 by adding x_k - x_k-1, which is known as the LOPCG correction and was suggested by Money and Ye. With krylovdim = 2, this algorithm becomes equivalent to LOPCG.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#Specific-algorithms-for-generalized-eigenvalue-problems-1",
    "page": "Available algorithms",
    "title": "Specific algorithms for generalized eigenvalue problems",
    "category": "section",
    "text": "GolubYe"
},

{
    "location": "man/algorithms/#KrylovKit.GKL",
    "page": "Available algorithms",
    "title": "KrylovKit.GKL",
    "category": "type",
    "text": "GKL(; krylovdim = KrylovDefaults.krylovdim, maxiter = KrylovDefaults.maxiter,\n    tol = KrylovDefaults.tol, orth = KrylovDefaults.orth, verbosity = 0)\n\nRepresents the Golub-Kahan-Lanczos bidiagonalization algorithm for sequentially building a Krylov-like factorization of a genereal matrix or linear operator with a bidiagonal reduced matrix. Can be used in svdsolve. The corresponding algorithm builds a Krylov subspace of size at most krylovdim, which will be repeated at most maxiter times and will stop when the norm of the residual of the Arnoldi factorization is smaller than tol. The orthogonalizer orth will be used to orthogonalize the different Krylov vectors. Default verbosity level verbosity is zero, meaning that no output will be printed.\n\nSee also: svdsolve, Orthogonalizer\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#Specific-algorithms-for-sigular-value-problems-1",
    "page": "Available algorithms",
    "title": "Specific algorithms for sigular value problems",
    "category": "section",
    "text": "GKL"
},

{
    "location": "man/algorithms/#KrylovKit.KrylovDefaults",
    "page": "Available algorithms",
    "title": "KrylovKit.KrylovDefaults",
    "category": "module",
    "text": "module KrylovDefaults\n    const orth = KrylovKit.ModifiedGramSchmidtIR()\n    const krylovdim = 30\n    const maxiter = 100\n    const tol = 1e-12\nend\n\nA module listing the default values for the typical parameters in Krylov based algorithms:\n\north: the orthogonalization routine used to orthogonalize the Krylov basis in the   Lanczos or Arnoldi iteration\nkrylovdim: the maximal dimension of the Krylov subspace that will be constructed\nmaxiter: the maximal number of outer iterations, i.e. the maximum number of times the   Krylov subspace may be rebuilt\ntol: the tolerance to which the problem must be solved, based on a suitable error   measure, e.g. the norm of some residual.\n\nwarning: Warning\nThe default value of tol is a Float64 value, if you solve problems in Float32 or ComplexF32 arithmetic, you should always specify a new tol as the default value will not be attainable.\n\n\n\n\n\n"
},

{
    "location": "man/algorithms/#Default-values-1",
    "page": "Available algorithms",
    "title": "Default values",
    "category": "section",
    "text": "KrylovDefaults"
},

{
    "location": "man/implementation/#",
    "page": "Details of the implementation",
    "title": "Details of the implementation",
    "category": "page",
    "text": ""
},

{
    "location": "man/implementation/#Details-of-the-implementation-1",
    "page": "Details of the implementation",
    "title": "Details of the implementation",
    "category": "section",
    "text": ""
},

{
    "location": "man/implementation/#KrylovKit.Basis",
    "page": "Details of the implementation",
    "title": "KrylovKit.Basis",
    "category": "type",
    "text": "abstract type Basis{T} end\n\nAn abstract type to collect specific types for representing a basis of vectors of type T.\n\nImplementations of Basis{T} behave in many ways like Vector{T} and should have a length, can be indexed (getindex and setindex!), iterated over (iterate), and support resizing (resize!, pop!, push!, empty!, sizehint!).\n\nThe type T denotes the type of the elements stored in an Basis{T} and can be any custom type that has vector like behavior (as defined in the docs of KrylovKit).\n\nSee OrthonormalBasis.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.OrthonormalBasis",
    "page": "Details of the implementation",
    "title": "KrylovKit.OrthonormalBasis",
    "category": "type",
    "text": "OrthonormalBasis{T} <: Basis{T}\n\nA list of vector like objects of type T that are mutually orthogonal and normalized to one, representing an orthonormal basis for some subspace (typically a Krylov subspace). See also Basis\n\nOrthonormality of the vectors contained in an instance b of OrthonormalBasis (i.e. all(dot(b[i],b[j]) == I[i,j] for i=1:lenght(b), j=1:length(b))) is not checked when elements are added; it is up to the algorithm that constructs b to guarantee orthonormality.\n\nOne can easily orthogonalize or orthonormalize a given vector v with respect to a b::OrthonormalBasis using the functions w, = orthogonalize(v,b,...) or w, = orthonormalize(v,b,...). The resulting vector w of the latter can then be added to b using push!(b, w). Note that in place versions orthogonalize!(v, b, ...) or orthonormalize!(v, b, ...) are also available.\n\nFinally, a linear combination of the vectors in b::OrthonormalBasis can be obtained by multiplying b with a Vector{<:Number} using * or mul! (if the output vector is already allocated).\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.orthogonalize",
    "page": "Details of the implementation",
    "title": "KrylovKit.orthogonalize",
    "category": "function",
    "text": "orthogonalize(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, x\northogonalize!(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, x\n\northogonalize(v, q, algorithm::Orthogonalizer]) -> w, s\northogonalize!(v, q, algorithm::Orthogonalizer]) -> w, s\n\nOrthogonalize vector v against all the vectors in the orthonormal basis b using the orthogonalization algorithm algorithm, and return the resulting vector w and the overlap coefficients x of v with the basis vectors in b.\n\nIn case of orthogonalize!, the vector v is mutated in place. In both functions, storage for the overlap coefficients x can be provided as optional argument x::AbstractVector with length(x) >= length(b).\n\nOne can also orthogonalize v against a given vector q (assumed to be normalized), in which case the orthogonal vector w and the inner product s between v and q are returned.\n\nNote that w is not normalized, see also orthonormalize.\n\nFor algorithms, see ClassicalGramSchmidt, ModifiedGramSchmidt, ClassicalGramSchmidt2, ModifiedGramSchmidt2, ClassicalGramSchmidtIR and ModifiedGramSchmidtIR.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.orthonormalize",
    "page": "Details of the implementation",
    "title": "KrylovKit.orthonormalize",
    "category": "function",
    "text": "orthonormalize(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, β, x\northonormalize!(v, b::OrthonormalBasis, [x::AbstractVector,] algorithm::Orthogonalizer]) -> w, β, x\n\northonormalize(v, q, algorithm::Orthogonalizer]) -> w, β, s\northonormalize!(v, q, algorithm::Orthogonalizer]) -> w, β, s\n\nOrthonormalize vector v against all the vectors in the orthonormal basis b using the orthogonalization algorithm algorithm, and return the resulting vector w (of norm 1), its norm β after orthogonalizing and the overlap coefficients x of v with the basis vectors in b, such that v = β * w + b * x.\n\nIn case of orthogonalize!, the vector v is mutated in place. In both functions, storage for the overlap coefficients x can be provided as optional argument x::AbstractVector with length(x) >= length(b).\n\nOne can also orthonormalize v against a given vector q (assumed to be normalized), in which case the orthonormal vector w, its norm β before normalizing and the inner product s between v and q are returned.\n\nSee orthogonalize if w does not need to be normalized.\n\nFor algorithms, see ClassicalGramSchmidt, ModifiedGramSchmidt, ClassicalGramSchmidt2, ModifiedGramSchmidt2, ClassicalGramSchmidtIR and ModifiedGramSchmidtIR.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#Orthogonalization-1",
    "page": "Details of the implementation",
    "title": "Orthogonalization",
    "category": "section",
    "text": "To denote a basis of vectors, e.g. to represent a given Krylov subspace, there is an abstract type Basis{T}KrylovKit.BasisMany Krylov based algorithms use an orthogonal basis to parametrize the Krylov subspace. In that case, the specific implementation OrthonormalBasis{T} can be used:KrylovKit.OrthonormalBasis{T}We can orthogonalize or orthonormalize a given vector to another vector (assumed normalized) or to a given OrthonormalBasis.KrylovKit.orthogonalize\nKrylovKit.orthonormalize"
},

{
    "location": "man/implementation/#Dense-linear-algebra-1",
    "page": "Details of the implementation",
    "title": "Dense linear algebra",
    "category": "section",
    "text": "KrylovKit relies on Julia\'s LinearAlgebra module from the standard library for most of its dense linear algebra dependencies."
},

{
    "location": "man/implementation/#KrylovKit.KrylovFactorization",
    "page": "Details of the implementation",
    "title": "KrylovKit.KrylovFactorization",
    "category": "type",
    "text": "abstract type KrylovFactorization{T,S<:Number}\nmutable struct LanczosFactorization{T,S<:Real}    <: KrylovFactorization{T,S}\nmutable struct ArnoldiFactorization{T,S<:Number}  <: KrylovFactorization{T,S}\n\nStructures to store a Krylov factorization of a linear map A of the form\n\n    A * V = V * B + r * b\'.\n\nFor a given Krylov factorization fact of length k = length(fact), the basis A is obtained via [basis(fact)](@ref basis) and is an instance of some subtype of [Basis{T}](@ref Basis), with alsolength(V) == kand whereTdenotes the type of vector like objects used in the problem. The Rayleigh quotientBis obtained as [rayleighquotient(fact)](@ref) andtypeof(B)is some subtype ofAbstractMatrix{S}withsize(B) == (k,k), typically a structured matrix. The residualris obtained as [residual(fact)](@ref) and is of typeT. One can also query [normres(fact)](@ref) to obtainnorm(r), the norm of the residual. The vectorbhas no dedicated name and often takes a default form (see below). It should be a subtype ofAbstractVectorof lengthkand can be obtained as [rayleighextension(fact)`](@ref) (by lack of a better dedicated name).\n\nIn particular, LanczosFactorization stores a Lanczos factorization of a real symmetric or complex hermitian linear map and has V::OrthonormalBasis{T} and B::SymTridiagonal{S<:Real}. ArnoldiFactorization stores an Arnoldi factorization of a general linear map and has V::OrthonormalBasis{T} and B::PackedHessenberg{S<:Number}. In both cases, b takes the default value e_k, i.e. the unit vector of all zeros and a one in the last entry, which is represented using SimpleBasisVector.\n\nA Krylov factorization fact can be destructured as V, B, r, nr, b = fact with nr = norm(r).\n\nLanczosFactorization and ArnoldiFactorization are mutable because they can expand! or shrink!. See also KrylovIterator (and in particular LanczosIterator and ArnoldiIterator) for iterators that construct progressively expanding Krylov factorizations of a given linear map and a starting vector.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.basis",
    "page": "Details of the implementation",
    "title": "KrylovKit.basis",
    "category": "function",
    "text": "    basis(fact::KrylovFactorization)\n\nReturn the list of basis vectors of a KrylovFactorization, which span the Krylov subspace. The return type is a subtype of Basis{T}, where T represents the type of the vectors used by the problem.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.rayleighquotient",
    "page": "Details of the implementation",
    "title": "KrylovKit.rayleighquotient",
    "category": "function",
    "text": "rayleighquotient(fact::KrylovFactorization)\n\nReturn the Rayleigh quotient of a KrylovFactorization, i.e. the reduced matrix within the basis of the Krylov subspace. The return type is a subtype of AbstractMatrix{<:Number}, typically some structured matrix type.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.residual",
    "page": "Details of the implementation",
    "title": "KrylovKit.residual",
    "category": "function",
    "text": "residual(fact::KrylovFactorization)\n\nReturn the residual of a KrylovFactorization. The return type is some vector of the same type as used in the problem. See also normres(F) for its norm, which typically has been computed already.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.normres",
    "page": "Details of the implementation",
    "title": "KrylovKit.normres",
    "category": "function",
    "text": "normres(fact::KrylovFactorization)\n\nReturn the norm of the residual of a KrylovFactorization. As this has typically already been computed, it is cheaper than (but otherwise equivalent to) norm(residual(F)).\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.rayleighextension",
    "page": "Details of the implementation",
    "title": "KrylovKit.rayleighextension",
    "category": "function",
    "text": "rayleighextension(fact::KrylovFactorization)\n\nReturn the vector b appearing in the definition of a KrylovFactorization.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#Krylov-factorizations-1",
    "page": "Details of the implementation",
    "title": "Krylov factorizations",
    "category": "section",
    "text": "The central ingredient in a Krylov based algorithm is a Krylov factorization or decomposition of a linear map. Such partial factorizations are represented as a KrylovFactorization, of which LanczosFactorization and ArnoldiFactorization are two concrete implementations:KrylovKit.KrylovFactorizationA KrylovFactorization can be destructered into its defining components using iteration, but these can also be accessed using the following functionsbasis\nrayleighquotient\nresidual\nnormres\nrayleighextension"
},

{
    "location": "man/implementation/#KrylovKit.KrylovIterator",
    "page": "Details of the implementation",
    "title": "KrylovKit.KrylovIterator",
    "category": "type",
    "text": "abstract type KrylovIterator{F,T}\nstruct LanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}\nstruct ArnoldiIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}\n\nLanczosIterator(f, v₀, [orth::Orthogonalizer = KrylovDefaults.orth], keepvecs::Bool = true)\nArnoldiIterator(f, v₀, [orth::Orthogonalizer = KrylovDefaults.orth])\n\nIterators that take a linear map of type F and an initial vector of type T and generate an expanding KrylovFactorization thereof.\n\nIn particular, for a real symmetric or complex hermitian linear map f, LanczosIterator uses the Lanczos iteration scheme to build a successively expanding LanczosFactorization. While f cannot be tested to be symmetric or hermitian directly when the linear map is encoded as a general callable object or function, it is tested whether the imaginary part of dot(v, f(v)) is sufficiently small to be neglected.\n\nSimilarly, for a general linear map f, ArnoldiIterator iterates over progressively expanding ArnoldiFactorizations using the Arnoldi iteration.\n\nThe optional argument orth specifies which Orthogonalizer to be used. The default value in KrylovDefaults is to use ModifiedGramSchmidtIR, which possibly uses reorthogonalization steps. For LanczosIterator, one can use to discard the old vectors that span the Krylov subspace by setting the final argument keepvecs to false. This, however, is only possible if an orth algorithm is used that does not rely on reorthogonalization, such as ClassicalGramSchmidt() or ModifiedGramSchmidt(). In that case, the iterator strictly uses the Lanczos three-term recurrence relation.\n\nWhen iterating over an instance of KrylovIterator, the values being generated are subtypes of KrylovFactorization, which can be immediately destructured into a basis, rayleighquotient, residual, normres and rayleighextension, for example as\n\nfor V,B,r,nr,b in ArnoldiIterator(f, v₀)\n    # do something\n    nr < tol && break # a typical stopping criterion\nend\n\nNote, however, that if keepvecs=false in LanczosIterator, the basis V cannot be extracted. Since the iterators don\'t know the dimension of the underlying vector space of objects of type T, they keep expanding the Krylov subspace until normres falls below machine precision eps for the given eltype(T).\n\nThe internal state of LanczosIterator and ArnoldiIterator is the same as the return value, i.e. the corresponding LanczosFactorization or ArnoldiFactorization. However, as Julia\'s Base iteration interface (using Base.iterate) requires that the state is not mutated, a deepcopy is produced upon every next iteration step.\n\nInstead, you can also mutate the KrylovFactorization in place, using the following interface, e.g. for the same example above\n\niterator = ArnoldiIterator(f, v₀)\nfactorization = initialize(iterator)\nwhile normres(factorization) > tol\n    expand!(iterator, f)\n    V,B,r,nr,b = factorization\n    # do something\nend\n\nHere, initialize(::KrylovIterator) produces the first Krylov factorization of length 1, and expand!(::KrylovIterator,::KrylovFactorization)(@ref) expands the factorization in place. See also initialize!(::KrylovIterator,::KrylovFactorization) to initialize in an already existing factorization (most information will be discarded) and shrink!(::KrylovFactorization, k) to shrink an existing factorization down to length k.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.expand!",
    "page": "Details of the implementation",
    "title": "KrylovKit.expand!",
    "category": "function",
    "text": "expand!(iter::KrylovIteraotr, fact::KrylovFactorization)\n\nExpand the Krylov factorization fact by one using the linear map and parameters in iter.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.shrink!",
    "page": "Details of the implementation",
    "title": "KrylovKit.shrink!",
    "category": "function",
    "text": "shrink!(fact::KrylovFactorization, k)\n\nShrink an existing Krylov factorization fact down to have length k. Does nothing if length(fact)<=k.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.initialize!",
    "page": "Details of the implementation",
    "title": "KrylovKit.initialize!",
    "category": "function",
    "text": "initialize!(iter::KrylovIteraotr, fact::KrylovFactorization)\n\nInitialize a length 1 Kryov factorization corresponding to iter in the already existing factorization fact, thereby destroying all the information it currently holds.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#KrylovKit.initialize",
    "page": "Details of the implementation",
    "title": "KrylovKit.initialize",
    "category": "function",
    "text": "initialize(iter::KrylovIteraotr)\n\nInitialize a length 1 Kryov factorization corresponding to iter.\n\n\n\n\n\n"
},

{
    "location": "man/implementation/#Krylov-iterators-1",
    "page": "Details of the implementation",
    "title": "Krylov iterators",
    "category": "section",
    "text": "Given a linear map A and a starting vector x₀, a Krylov factorization is obtained by sequentially building a Krylov subspace x₀ A x₀ A² x₀ . Rather then using this set of vectors as a basis, an orthonormal basis is generated by a process known as Lanczos or Arnoldi iteration (for symmetric/hermitian and for general matrices, respectively). These processes are represented as iterators in Julia:KrylovKit.KrylovIteratorThe following functions allow to manipulate a KrylovFactorization obtained from such a KrylovIterator:expand!\nshrink!\ninitialize!\ninitialize"
},

]}
