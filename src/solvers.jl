
function powermethod(A::Function, b::AbstractArray; maxit=100, tol=1e-3, verbose=true)
    r = zeros(maxit)
    λ, λᵏ= 0, 0
    flag = true         # error flag: tolerance not reached
    for k ∈ 1:maxit
        # b = A(b)
        # b = b ./ norm(b)
        # λ = sum(conj(b).*A(b))
        Ab = A(b)
        λ = sum(conj(b).*Ab) / sum(abs2, b)
        b = Ab ./ norm(b)
        r[k] = abs(λ-λᵏ)
        λᵏ = λ
        if verbose
            @printf "k: %3d, |λ-λᵏ|= %.3e\n" k r[k]
        end
        if r[k] <= tol
            flag = false; 
            break
        end
    end
    return λ, b, flag
end

