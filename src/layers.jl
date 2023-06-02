
struct Threshold{F} <: Lux.AbstractExplicitLayer
    dims::Tuple
    init_thresh::Float32
    init_states
end

function tensor_evalpoly(x, coeffs::AbstractArray{T,N}) where{T, N}
    p = selectdim(coeffs, N, 1) .* x.^0
    for d in 1:size(coeffs, N)-1
        p = p + selectdim(coeffs, N, d+1) .* x.^d
    end
    return p
end

function init_poly(rng::AbstractRNG, dims, t0)
    weight = zeros(Float32, dims)
    selectdim(weight, length(dims), 1) .= Float32(t0)
    return weight
end

Lux.initialparameters(rng::AbstractRNG, l::Threshold) = (weight=init_poly(rng, l.dims, l.init_thresh),)
Lux.initialstates(rng::AbstractRNG, l::Threshold) = l.init_states()

# soft thresholding
soft_thresh(x, t) = sign(x)*relu(abs(x) - t)
const SoftThresh = Threshold{typeof(soft_thresh)} 

function SoftThresh(degree::Int, channels::Int, t0::Float32) 
    return SoftThresh((1,1,channels,degree+1), t0, ()->NamedTuple())
end

function (l::SoftThresh)((x, σ)::Tuple, ps, st::NamedTuple)
    τ = tensor_evalpoly(σ, ps.weight)
    return soft_thresh.(x, τ), st
end

struct PGMLayer{C, Ct, T} <: Lux.AbstractExplicitContainerLayer{(:analysis, :synthesis, :prox)}
    analysis::C
    synthesis::Ct
    prox::T
end

const LISTALayer = PGMLayer{C, Ct, <: SoftThresh} where {C, Ct}

function (l::LISTALayer)((y, H)::Tuple{<:AbstractArray, <:AbstractOperator}, ps, st::NamedTuple) 
    Ar, st_a = l.analysis(H'(y), ps.analysis, st.analysis)
    z,  st_p = l.prox((Ar, H.noise_level), ps.prox, st.prox)
    return z, (analysis=st_a, synthesis=st.synthesis, prox=st_p, z=z)
end

function (l::LISTALayer)((z, y, H)::Tuple{T, T, <:AbstractOperator}, ps, st::NamedTuple) where {T <: AbstractArray}
    Bz, st_s = l.synthesis(z, ps.synthesis, st.synthesis)
    Ar, st_a = l.analysis(gramian(H)(Bz) - H'(y), ps.analysis, st.analysis)
    z, st_p  = l.prox((z - Ar, H.noise_level), ps.prox, st.prox)
    return z, (analysis=st_a, synthesis=st_s, prox=st_p, z=z)
end

struct LISTA{K, T <: LISTALayer} <: Lux.AbstractExplicitLayer
    layer::T
end

function (l::LISTA{K})((y, H)::Tuple, ps, st::NamedTuple) where {K}
    z, st = l.layer((y, H), ps[1], st)
    for k in 2:K
        z, st = l.layer((z, y, H), ps[k], st)
    end
    return z, st
end

function Lux.initialparameters(rng::AbstractRNG, l::LISTA{K}) where {K}
    names = ntuple(k->Symbol("layer_$k"), Val(K))
    params = ntuple(k->Lux.initialparameters(rng, l.layer), Val(K))
    return NamedTuple{names}(params)
end

function Lux.initialstates(rng::AbstractRNG, l::LISTA)
    return Lux.initialstates(rng, l.layer)
end

function spectral_setup(rng::AbstractRNG, l::LISTA{K}; device=Lux.cpu) where {K}
    ps_, st_ = spectral_setup(rng, l.layer; device=device)
    names = ntuple(k->Symbol("layer_$k"), Val(K))
    params = ntuple(k->deepcopy(ps_), Val(K))
    ps = NamedTuple{names}(params)
    return ps, st_
end

function spectral_setup(rng::AbstractRNG, l::PGMLayer; device=Lux.cpu) 
    ps, st = Lux.setup(rng, l)
    chain = Lux.Chain(l.analysis, l.synthesis)
    ps_ = (layer_1=ps.analysis, layer_2=ps.synthesis) |> device
    st_ = (layer_1=st.analysis, layer_2=st.synthesis) |> device
    @reset ps_.layer_2 = deepcopy(ps_.layer_1)

    @assert l.analysis isa Union{Lux.Conv, GaborConv}
    if l.analysis isa Lux.Conv
        @reset ps_.layer_2.weight = conj(ps_.layer_2.weight)
    elseif l.analysis isa GaborConv
        @reset ps_.layer_2.frequency = conj(ps_.layer_2.frequency)
        @reset ps_.layer_2.phase = conj(ps_.layer_2.phase)
    end

    BA(x) = first(chain(x, ps_, st_))
    in_chs = l.analysis isa GaborConv ? l.analysis.layer.in_chs : l.analysis.in_chs
    b = randn_like(rng, kernel(chain[1], ps_[1], st_[1]), (128, 128, in_chs, 1))
    L, _, flag = powermethod(BA, b; maxit=5000, tol=1e-4, verbose=false)

    @assert real(L) > 0 && imag(L) < 1e-3
    flag && @warn "spectral_init: power_method tolerance not reached."
    @info "spectral_init: power_method: L=$(L)"
    L = real(L)

    ps_ = ps_ |> device
    st_ = st_ |> device

    if l.analysis isa Lux.Conv
        W = ps_.layer_1.weight
        @reset ps.analysis.weight = copy(W) ./ sqrt(L)
        @reset ps.synthesis.weight = conj(copy(W)) ./ sqrt(L)
    elseif l.analysis isa GaborConv
        @reset ps.analysis  = deepcopy(ps_.layer_1)
        @reset ps.synthesis = deepcopy(ps_.layer_2)
        ps.analysis.scale  ./= sqrt(L)
        ps.synthesis.scale ./= sqrt(L)
    end
    
    return ps, st
end

