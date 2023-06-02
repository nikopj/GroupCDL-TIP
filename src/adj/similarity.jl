
function distance_similarity(x::AbstractArray{T, 4}, y::AbstractArray{T, 4}) where {T}
    szx, szy = size(x), size(y)
    xr = reshape(x, prod(szx[1:2]), szx[3], :)
    yr = permutedims(reshape(y, prod(szy[1:2]), szy[3], :), (2, 1, 3))
    return -relu(sum(abs2, xr, dims=2) .+ sum(abs2, yr, dims=1) .- T(2) .* (xr ⊠ yr)) / T(2)
end

function dot_similarity(x::AbstractArray{T, 4}, y::AbstractArray{T, 4}) where {T}
    szx, szy = size(x), size(y)
    xr = reshape(x, prod(szx[1:2]), szx[3], :)
    yr = reshape(y, prod(szy[1:2]), szy[3], :) 
    yr = permutedims(yr, (2,1,3))
    return (xr ⊠ yr) ./ sqrt(T(szx[3]))
end

function distance_similarity(x::AbstractArray{T,1}, y::AbstractArray{T,1}) where T
    s = zero(T)
    d = length(x)
    for i in 1:d
        s -= abs2(x[i]-y[i])
    end
    return s / T(2)
end

function dot_similarity(x::AbstractArray{T,1}, y::AbstractArray{T,1}) where T
    s = zero(T)
    d = length(x)
    for i in 1:d
        s += x[i]*y[i]
    end
    return s / sqrt(T(d))
end

function sparse_similarity!(S::AbstractSparseMatrix, simfun, x, y, ws)
    @assert size(x, 4) == 1 "batchsize=$(size(x,4)) must be 1"
    I, J, _ = findnz(S)
    C = CartesianIndices(size(x)[1:2])

    Threads.@threads for k=1:length(I)
        i, j = I[k], J[k]
        S[i,j] = simfun(x[C[i], :, 1], y[C[j], :, 1]) 
    end
    return S
end

function sparse_similarity!(S::CuSparseMatrixCSR, simfun, x, y, W)
    @assert size(x, 4) == 1 "batchsize=$(size(x,4)) must be 1"
    maxidx = S.nnz 
    args = S, simfun, x, y, W, maxidx
    kernel = @cuda launch=false sparse_similarity_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return S 
end

function sparse_similarity_kernel!(S::AbstractArray{T}, ::typeof(distance_similarity), x, y, W, maxidx) where T
    n = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if n <= maxidx
        N1, N2, M = size(x)[1:3]
        C = CartesianIndices((N1, N2))
        i, j = cartesian_circulant(n, N1, N2, W)
        Ci, Cj = C[i], C[j]
        s = zero(T)
        for m=1:M
            s -= abs2(x[Cj, m, 1] - y[Ci, m, 1])
        end
        S.nzVal[n] = s / T(2)
    end
    return nothing
end

function sparse_similarity_kernel!(S::AbstractArray{T}, ::typeof(dot_similarity), x, y, W, maxidx) where T
    n = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if n <= maxidx
        N1, N2, M = size(x)[1:3]
        C = CartesianIndices((N1, N2))
        i, j = cartesian_circulant(n, N1, N2, W)
        Ci, Cj = C[i], C[j]
        s = zero(T)
        for m=1:M
            s += x[Cj, m, 1]*y[Ci, m, 1]
        end
        S.nzVal[n] = s / sqrt(T(M))
    end
    return nothing
end

