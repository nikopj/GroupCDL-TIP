CuSparseMatrixCSC{Tv, Ti}(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = A
CuSparseMatrixCSC{Tv, Ti}(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixCSC(A)
CuSparseMatrixCSR{Tv, Ti}(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti} = A
CuSparseMatrixCSR{Tv, Ti}(A::CuSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = CuSparseMatrixCSR(A)

function sumdim1_kernel!(f, Z, A, maxidx)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if idx <= maxidx
        for k in A.colPtr[idx]:A.colPtr[idx+1]-1
            Z[idx] += f(A.nzVal[k])
        end
    end
    return nothing
end

function sumdim1!(f, Z, A::CuSparseMatrixCSC)
    maxidx = size(A, 2)
    args = f, reshape(Z, :), A, maxidx
    kernel = @cuda launch=false sumdim1_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return Z
end
sumdim1(f, A::CuSparseMatrixCSC{T}) where {T} = sumdim1!(f, CUDA.zeros(T, 1, size(A,2)), A)
sumdim1(f, A::AbstractCuSparseMatrix) = sumdim1(f, CuSparseMatrixCSC(A))
sumdim1(A) = sumdim1(identity, A)

function sumdim2_kernel!(f, Z, A, maxidx)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if idx <= maxidx
        for k in A.rowPtr[idx]:A.rowPtr[idx+1]-1
            Z[idx] += f(A.nzVal[k])
        end
    end
    return nothing
end

function sumdim2!(f, Z, A::CuSparseMatrixCSR)
    maxidx = size(A, 1)
    args = f, reshape(Z, :), A, maxidx
    kernel = @cuda launch=false sumdim2_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return Z
end
sumdim2(f, A::CuSparseMatrixCSR{T}) where {T} = sumdim2!(f, CuMatrix{T}(undef, size(A, 2), 1), A)
sumdim2(f, A::AbstractCuSparseMatrix) = sumdim2(f, CuSparseMatrixCSR(A))
sumdim2(A) = sumdim2(identity, A)

function softmax_kernel!(T, nzval, ptr, maxidx)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    idx > maxidx && return

    nzrange = (ptr[idx]):(ptr[idx+1]-1)
    s = zero(T) + 1f-8
    @inbounds for k in nzrange
        nzval[k] = exp(nzval[k])
        s += nzval[k]
    end
    @inbounds for k in nzrange
        nzval[k] /= s
    end
    return nothing
end

function sparse_softmax!(A::T; dims=1) where {T <: AbstractCuSparseMatrix}
    if dims == 1
        A = CuSparseMatrixCSC(A) 
        nzval = A.nzVal
        ptr = A.colPtr
        maxidx = size(A, 2)
    elseif dims == 2
        A = CuSparseMatrixCSR(A) 
        nzval = A.nzVal
        ptr = A.rowPtr
        maxidx = size(A, 1)
    else
        throw("Invalid argument! dims=$dims must be 1 or 2.")
    end

    args = eltype(A), nzval, ptr, maxidx
    kernel = @cuda launch=false softmax_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return T(A)
end

function sparse_softmax_dim1!(A::SparseMatrixCSC{T}) where {T}
    N = size(A, 2)
    A.nzval .= exp.(A.nzval)
    Z = similar(A.nzval, N)
    Threads.@threads for c=1:N
        for k in nzrange(A, c)
            Z[c] += A.nzval[k]
        end
    end

    Threads.@threads for c=1:N
        for k in nzrange(A, c)
            A.nzval[k] /= Z[c]
        end
    end
    return A
end

function sparse_softmax_dim2!(A::SparseMatrixCSC{T}) where {T}
    N = size(A, 1)
    A.nzval .= exp.(A.nzval)
    Z = [Threads.Atomic{T}(zero(T)) for i=1:N]
    Threads.@threads for c=1:N
        for k in nzrange(A, c)
            r = A.rowval[k]
            Threads.atomic_add!(Z[r], A.nzval[k])
        end
    end

    Threads.@threads for c=1:N
        for k in nzrange(A, c)
            r = A.rowval[k]
            A.nzval[k] /= Z[r].value
        end
    end
    return A
end

function sparse_softmax!(A::SparseMatrixCSC; dims=1)
    if dims == 1
        A = sparse_softmax_dim1!(A)
    elseif dims == 2
        A = sparse_softmax_dim2!(A)
    else
        throw("Invalid argument! dims=$dims must be 1 or 2.")
    end
    return A
end

sparse_softmax(A::SparseMatrixCSC; dims=1) = sparse_softmax!(copy(A); dims=dims)
sparse_softmax(A::CuSparseMatrixCSR; dims=1) = sparse_softmax!(copy(A); dims=dims)

