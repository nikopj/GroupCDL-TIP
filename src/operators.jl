abstract type AbstractOperator end

struct AddNoise end

struct AWGN{S <: AbstractOperator, T <: Union{<:Real,<:AbstractArray}} <: AbstractOperator
    parent::S
    noise_level::T
end
struct AdjointOp{S <: AbstractOperator} <: AbstractOperator
    parent::S
end
struct Gramian{S <: AbstractOperator} <: AbstractOperator
    parent::S
end
const AdjointAbsOp = AdjointOp{<:AbstractOperator}
const GramianAbsOp = Gramian{<:AbstractOperator}
const AWGNAbsOp = AWGN{<:AbstractOperator}

Base.adjoint(A::AbstractOperator) = AdjointOp(A)
Base.adjoint(A::AdjointAbsOp) = A.parent
gramian(A::AbstractOperator) = Gramian(A)

awgn(A::AbstractOperator, σ) = AWGN(A, σ)
Base.adjoint(A::AWGNAbsOp) = AWGN(A.parent', A.noise_level)
gramian(A::AWGNAbsOp) = AWGN(gramian(A.parent), A.noise_level)
(A::AWGNAbsOp)(x) = A.parent(x)
(A::AWGNAbsOp)(x, ::AddNoise) = awgn(A.parent(x), A.noise_level)
(A::AWGNAbsOp)(rng::AbstractRNG, x, ::AddNoise) = awgn(rng, A.parent(x), A.noise_level)

struct Identity <: AbstractOperator end
(A::Identity)(x) = x
(A::AdjointOp{Identity})(x) = x
(A::Gramian{Identity})(x) = x

struct Fourier <: AbstractOperator end
(A::Fourier)(x) = fft(x, (1,2))
(A::AdjointOp{Fourier})(x) = ifft(x, (1,2))
(A::Gramian{Fourier})(x) = x

struct MaskFourier <: AbstractOperator
    mask
end
(A::MaskFourier)(x) = A.mask .* fft(x, (1,2))
(A::AdjointOp{MaskFourier})(x) = ifft(A.parent.mask .* x, (1,2)) 
(A::Gramian{MaskFourier})(x) = ifft(A.parent.mask .* fft(x, (1,2)), (1,2))

awgn(rng::AbstractRNG, x, σ) =  x + σ .* randn!(rng, similar(x))
awgn(x, σ) = x + σ .* randn!(similar(x))

@inline function rand_range(rng::AbstractRNG, batch::AbstractArray{T, N}, (a, b)::Union{Vector, Tuple}) where {T, N}
    return a .+ (b-a).*rand!(rng, similar(batch, (ones(Int, N-1)..., size(batch, N))))
end
rand_range(batch, t) = rand_range(Random.GLOBAL_RNG, batch, t)

