cisorcos(::Val{false}, x) = cospi(x)
cisorcos(::Val{true}, x)  = cispi(x)

function kernelgrid(ks::Int)
    # input: kernelsize
    # output: (2, ks, ks)
    m, n = ceil(Int, (ks-1)/2), floor(Int, (ks-1)/2)
    u = ones(ks)
    v = copy(u)
    v .= -m:n
    X = u * v'
    return stack((X, X'), dims=1)
end
CRC.@non_differentiable kernelgrid(::Any...)

function gaborkernel(a, f, ψ, grid::AbstractArray, is_complex::Val=Val(false))
    """
    a (precision): (2, 1, 1, in_ch, out_ch, batch)
    f (freq):      (2, 1, 1, in_ch, out_ch, batch)
    ψ (phase):        (1, 1, in_ch, out_ch, batch)
    output:   (ks, ks, in_ch, out_ch, batch)
    """
    ax = dropdims(sum(abs2, a.*grid, dims=1), dims=1)
    fx = dropdims(sum(f.*grid, dims=1), dims=1)
    return @. exp(-ax)*cisorcos(is_complex, fx + ψ)
end
gaborkernel(a, f, ψ, ks::Int, is_complex::Val=Val(false)) = gaborkernel(a, f, ψ, kernelgrid(ks), is_complex)

const ConvOrConvT = Union{Lux.Conv, Lux.ConvTranspose}

struct GaborConvLayer{L <: ConvOrConvT} <: Lux.AbstractExplicitLayer
    layer::L
    MoG::Int
    is_complex::Val
end

const GaborConv = GaborConvLayer{<: Lux.Conv}
const GaborConvTranspose = GaborConvLayer{<: Lux.ConvTranspose}

function GaborConv(args...; is_complex=false, MoG=1, kws...)
    C = Lux.Conv(args...; kws...)
    return GaborConvLayer{typeof(C)}(C, MoG, Val(is_complex))
end

function GaborConvTranspose(args...; is_complex=false, MoG=1, kws...)
    C = Lux.ConvTranspose(args...; kws...)
    return GaborConvLayer{typeof(C)}(C, MoG, Val(is_complex))
end

function Lux.initialparameters(rng::AbstractRNG, g::GaborConvLayer)
    ps = (
        scale     = randn(rng, Float32, 1, 1, g.layer.in_chs, g.layer.out_chs, g.MoG),
        precision = randn(rng, Float32, 2, 1, 1, g.layer.in_chs, g.layer.out_chs, g.MoG),
        frequency = randn(rng, Float32, 2, 1, 1, g.layer.in_chs, g.layer.out_chs, g.MoG),
        phase     = randn(rng, Float32, 1, 1, g.layer.in_chs, g.layer.out_chs, g.MoG)
    )
    return ps
end

Lux.initialstates(::AbstractRNG, g::GaborConvLayer) = (kernelgrid=kernelgrid(g.layer.kernel_size[1]),)

function Lux.parameterlength(g::GaborConvLayer)
    return 6*g.MoG*g.layer.in_chs*g.layer.out_chs
end

@inline function kernel(g::GaborConvLayer, ps, st)
    W = ps.scale .* gaborkernel(ps.precision, ps.frequency, ps.phase, st.kernelgrid, g.is_complex)
    return dropdims(sum(W, dims=5), dims=5)
end

@inline function (g::GaborConvLayer)(x, ps, st) 
   return g.layer(x, (weight=kernel(g, ps, st),), st)
end

@inline function kernel(l::Union{Lux.Conv, Lux.ConvTranspose}, ps, st)
    return ps.weight
end

