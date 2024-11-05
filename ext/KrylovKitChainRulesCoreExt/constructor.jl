function ChainRulesCore.rrule(::Type{RecursiveVec}, A) 
    function RecursiveVec_pullback(ΔA)
        return NoTangent(), ΔA.vecs
    end
    return RecursiveVec(A), RecursiveVec_pullback
end
