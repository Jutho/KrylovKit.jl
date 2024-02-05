# the following definition is used to compare sets of eigenvalues
function ≊(list1::AbstractVector, list2::AbstractVector)
    length(list1) == length(list2) || return false
    n = length(list1)
    ind2 = collect(1:n)
    p = sizehint!(Int[], n)
    for i in 1:n
        j = argmin(abs.(view(list2, ind2) .- list1[i]))
        p = push!(p, ind2[j])
        ind2 = deleteat!(ind2, j)
    end
    return list1 ≈ view(list2, p)
end
