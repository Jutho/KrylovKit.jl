# reference implementation https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/master/src/lsmr.jl
linsolve(operator, b, alg::LSMR) = linsolve(operator,b,svdfun(operator)(x,true),alg);
function linsolve(operator, b, x, alg::LSMR)
    u = axpby!(1,b,-1,svdfun(operator)(x,false))
    β = norm(u);

    # initialize GKL factorization
    iter = GKLIterator(svdfun(operator), u, alg.orth)
    fact = initialize(iter; verbosity = alg.verbosity-2)
    numops = 2
    sizehint!(fact, alg.krylovdim)

    T = eltype(fact);
    Tr = real(T)
    alg.conlim > 0 ? ctol = convert(Tr, inv(alg.conlim)) : ctol = zero(Tr);
    istop = 0;

    for topit = 1:alg.maxiter# the outermost restart loop
        # Initialize variables for 1st iteration.
        α = fact.αs[end];
        ζbar = α * β
        αbar = α
        ρ = one(Tr)
        ρbar = one(Tr)
        cbar = one(Tr)
        sbar = zero(Tr)

        # Initialize variables for estimation of ||r||.
        βdd = β
        βd = zero(Tr)
        ρdold = one(Tr)
        τtildeold = zero(Tr)
        θtilde  = zero(Tr)
        ζ = zero(Tr)
        d = zero(Tr)

        # Initialize variables for estimation of ||A|| and cond(A).
        normA, condA, normx = -one(Tr), -one(Tr), -one(Tr)
        normA2 = abs2(α)
        maxrbar = zero(Tr)
        minrbar = 1e100

        # Items for use in stopping rules.
        normb = β
        normr = β
        normAr = α * β

        hbar = zero(T)*x;
        h = one(T)*fact.V[end];

        while length(fact) < alg.krylovdim

            β = normres(fact);
            fact = expand!(iter, fact)
            numops += 2;

            v = fact.V[end];
            α = fact.αs[end];

            # Construct rotation Qhat_{k,2k+1}.
            αhat = hypot(αbar, alg.λ)
            chat = αbar / αhat
            shat = alg.λ / αhat

            # Use a plane rotation (Q_i) to turn B_i to R_i.
            ρold = ρ
            ρ = hypot(αhat, β)
            c = αhat / ρ
            s = β / ρ
            θnew = s * α
            αbar = c * α

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
            ρbarold = ρbar
            ζold = ζ
            θbar = sbar * ρ
            ρtemp = cbar * ρ
            ρbar = hypot(cbar * ρ, θnew)
            cbar = cbar * ρ / ρbar
            sbar = θnew / ρbar
            ζ = cbar * ζbar
            ζbar = - sbar * ζbar

            # Update h, h_hat, x.
            hbar = axpby!(1,h,-θbar * ρ / (ρold * ρbarold),hbar);
            h = axpby!(1,v,-θnew / ρ,h);
            x = axpy!(ζ / (ρ * ρbar),hbar,x)

            ##############################################################################
            ##
            ## Estimate of ||r||
            ##
            ##############################################################################

            # Apply rotation Qhat_{k,2k+1}.
            βacute = chat * βdd
            βcheck = - shat * βdd

            # Apply rotation Q_{k,k+1}.
            βhat = c * βacute
            βdd = - s * βacute

            # Apply rotation Qtilde_{k-1}.
            θtildeold = θtilde
            ρtildeold = hypot(ρdold, θbar)
            ctildeold = ρdold / ρtildeold
            stildeold = θbar / ρtildeold
            θtilde = stildeold * ρbar
            ρdold = ctildeold * ρbar
            βd = - stildeold * βd + ctildeold * βhat

            τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
            τd = (ζ - θtilde * τtildeold) / ρdold
            d += abs2(βcheck)
            normr = sqrt(d + abs2(βd - τd) + abs2(βdd))

            # Estimate ||A||.
            normA2 += abs2(β)
            normA  = sqrt(normA2)
            normA2 += abs2(α)

            # Estimate cond(A).
            maxrbar = max(maxrbar, ρbarold)
            if length(fact) > 1
                minrbar = min(minrbar, ρbarold)
            end
            condA = max(maxrbar, ρtemp) / min(minrbar, ρtemp)


            ##############################################################################
            ##
            ## Test for convergence
            ##
            ##############################################################################

            # Compute norms for convergence testing.
            normAr  = abs(ζbar)
            normx = norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.
            test1 = normr / normb
            test2 = normAr / (normA * normr)
            test3 = inv(condA)

            t1 = test1 / (one(Tr) + normA * normx / normb)
            rtol = alg.btol + alg.atol * normA * normx / normb
            # The following tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            if alg.verbosity > 2
                msg = "LSMR linsolve in iter $topit; step $(length(fact)-1): "
                msg *= "normres = "
                msg *= @sprintf("%.12e", normr)
                @info msg
            end

            if 1 + test3 <= 1 istop = 6; break end
            if 1 + test2 <= 1 istop = 5; break end
            if 1 + t1 <= 1 istop = 4; break end
            # Allow for tolerances set by the user.
            if test3 <= ctol istop = 3; break end
            if test2 <= alg.atol istop = 2; break end
            if test1 <= rtol  istop = 1; break end
        end

        u = axpby!(1,b,-1,svdfun(operator)(x,false))

        istop != 0 && break;

        #restart
        β = norm(u);
        iter = GKLIterator(svdfun(operator), u, alg.orth);
        fact = initialize!(iter,fact);
    end

    isconv = istop ∉ (0,3,6);
    if alg.verbosity > 0 && !isconv
        @warn """LSMR linsolve finished without converging after $(alg.maxiter) iterations:
         *  norm of residual = $(norm(u))
         *  number of operations = $numops"""
     elseif alg.verbosity > 0
         if alg.verbosity > 0
             @info """LSMR linsolve converged due to istop $(istop):
              *  norm of residual = $(norm(u))
              *  number of operations = $numops"""
         end
    end
    return (x, ConvergenceInfo(Int(isconv), u, norm(u), alg.maxiter, numops))

end
