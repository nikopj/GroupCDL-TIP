"""
Construct a non-binary adjacency matrix Γ based on a sliding window neighborhood 
of `x` with windowsize `ws` and Γ = softmax(simfun(x[i], x[j]))
where Z[j] is such that Γ is column normalized.
"""
function sparse_adjacency(simfun, x::AbstractArray, y::AbstractArray, ws::Int)
    S = circulant(x, ws)
    S = sparse_similarity!(S, simfun, x, y, ws)
    return circulant_softmax!(S)
end

function sparse_adjacency!(Γ::AbstractSparseMatrix, simfun, x, y, ws)
    S = sparse_similarity!(Γ, simfun, x, y, ws)
    return circulant_softmax!(S)
end

# trainmode and uninitialized
function adjacency(::Missing, ::Missing, simfun, x, y, ws::Int, γ, ::Val{true})
    Γ = softmax(simfun(x, y), dims=2)
    return Γ, Γ
end

# trainmode and initialized
function adjacency(Γ, Γk, simfun, x, y, ws::Int, γ, ::Val{true})
    Γk = Γ
    Γ  = softmax(simfun(x, y), dims=2)
    return @. γ*Γ + (1f0 - γ)*Γk, Γk
end

# testmode and unitialized
function adjacency(::Missing, ::Missing, simfun, x, y, ws::Int, γ, ::Val{false})
    # we know that running_sim is a scalar zero when buffer is missing
    Γ = sparse_adjacency(simfun, x, y, ws)
    return Γ, deepcopy(Γ)
end

# testmode and initialized
function adjacency(Γ::AbstractSparseMatrix, Γk::AbstractSparseMatrix, simfun, x, y, ws::Int, γ, ::Val{false})
    Γk.nzval .= Γ.nzval
    Γ = sparse_adjacency!(Γ, simfun, x, y, ws)
    @. Γ.nzval = γ*Γ.nzval + (1f0 - γ)*Γk.nzval
    return Γ, Γk
end

# testmode and initialized
function adjacency(Γ::AbstractCuSparseMatrix, Γk::AbstractCuSparseMatrix, simfun, x, y, ws::Int, γ, ::Val{false})
    Γk.nzVal .= Γ.nzVal
    Γ = sparse_adjacency!(Γ, simfun, x, y, ws)
    @. Γ.nzVal = γ*Γ.nzVal + (1f0 - γ)*Γk.nzVal
    return Γ, Γk
end

