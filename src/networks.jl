abstract type AbstractPreprocess end

struct NaturalImagePreprocess <: AbstractPreprocess 
    stride::Int
end

function calcpad(N::Int, s::Int)
	p = s*ceil(N/s) - N
	return ceil(Int, p/2), floor(Int, p/2)
end

function calcpad(M::Int, N::Int, s::Int)
    return Tuple([calcpad(M, s)..., calcpad(N, s)...])
end
CRC.@non_differentiable calcpad(::Any...)

function unpad(x::AbstractArray, pad::NTuple{4, Int}) 
    return x[begin+pad[1]:end-pad[2], begin+pad[3]:end-pad[4], :, :]
end

function preprocess(P::NaturalImagePreprocess, y, ::AbstractOperator) 
    pad = calcpad(size(y,1), size(y,2), P.stride)
    yp = pad_reflect(y, pad, dims=(1,2))
    μ = mean(yp, dims=(1,2,3))
    return yp .- μ, (μ, pad)
end

function postprocess(::NaturalImagePreprocess, x, (μ, pad)::Tuple) 
    return unpad(x .+ μ, pad)
end

struct CDLNet{L, D, P <: AbstractPreprocess} <: Lux.AbstractExplicitContainerLayer{(:lista, :dictionary)}
    lista::L
    dictionary::D
    in_chs::Int
    subbands::Int
    stride::Int
    preproc::P
    is_complex::Bool
end

const GDLNet = CDLNet{L, <:GaborConvTranspose} where {L}
const GroupCDLNet = CDLNet{<: GroupLISTA}
randnC32(rng::AbstractRNG, size...) = randn(rng, ComplexF32, size...)

function CDLNet(;
        K::Int=20, 
        M::Int=32, 
        C::Int=1, 
        p::Int=7, 
        s::Int=1, 
        d::Int=0, 
        is_complex::Bool=false, 
        MoG::Int=0, 
        nlss_hidden::Int=M,
        nlss_similarity::String="distance",
        nlss_windowsize::Int=1, 
        nlss_Δupdate::Int=1,
        nlss_compressed=true,
    )
    padl, padr = ceil(Int,(p-s)/2), floor(Int,(p-s)/2)
    pad = (padl, padr, padl, padr)

    gabor = MoG > 0
    if gabor
        A = GaborConv((p,p), C=>M; is_complex=is_complex, MoG=MoG, pad=pad, stride=s, use_bias=false)
        B = GaborConvTranspose((p,p), M=>C; is_complex=is_complex, MoG=MoG, pad=pad, stride=s, use_bias=false)
    else
        init_weight = is_complex ? randnC32 : Lux.randn32
        A = Lux.Conv((p,p), C=>M; pad=pad, stride=s, init_weight=init_weight, use_bias=false)
        B = Lux.ConvTranspose((p,p), M=>C; pad=pad, stride=s, init_weight=init_weight, use_bias=false)
    end

    if nlss_windowsize > 1
        T = GroupThresh(d, M, 1f-2)
        layer = PGMLayer(A, B, T)
        nlss = NonlocalSelfSim(M => nlss_hidden, nlss_similarity, nlss_windowsize, 0.8f0, compressed=nlss_compressed)
        encoder = GroupLISTA{K, typeof(layer), typeof(nlss)}(layer, nlss, nlss_Δupdate)
    else
        T = SoftThresh(d, M, 1f-2)
        layer = PGMLayer(A, B, T)
        encoder = LISTA{K, typeof(layer)}(layer)
    end
    preproc = NaturalImagePreprocess(s)

    return CDLNet{typeof(encoder), typeof(B), typeof(preproc)}(encoder, B, C, M, s, preproc, is_complex)
end

function (n::CDLNet)((y, H)::Tuple, ps, st::NamedTuple) 
    y, ppt = preprocess(n.preproc, y, H)
    z, st_l = n.lista((y, H), ps.lista, st.lista)
    x, st_d = n.dictionary(z, ps.dictionary, st.dictionary)
    x = postprocess(n.preproc, x, ppt)
    return x, (lista=st_l, dictionary=st_d, csc=z)
end

function ow_unfold(x, W; stride=1)
    X = NNlib.unfold(x, (W, W, size(x, 3), 1); stride=stride, pad=(0, W-1, 0, W-1))
    X = permutedims(X, (2, 1, 3))
    return reshape(X, W, W, size(x, 3), :)
end

function ow_fold(X, szx, W; stride=1)
    X = reshape(X, W^2 * szx[3], :, szx[4])
    X = permutedims(X, (2, 1, 3))
    return NNlib.fold(X, szx, (W, W, szx[3], 1); stride=stride, pad=(0, W-1, 0, W-1))
end

function ow_forward(net, y, H, ps, st::NamedTuple; stride=10)
    # overlapping window forward-pass
    W = st.lista.nlss.windowsize*net.stride
    Y = ow_unfold(y, W; stride=stride)
    D = ow_unfold(ones_like(y), W; stride=stride)

    X = similar(Y)
    bs = 1
    dl = DataLoader(Y; batchsize=bs, collate=true)
    i = 1
    for Yb in dl
        Xb, _ = net((Yb, H), ps, st)
        X[:,:,:,i:i+size(Yb,4)-1] .= Xb
        i += size(Yb, 4) 
    end
    # uncomment/comment for -^-batched-^- or -v-parallel-v-
    # X, _ = net((Y, H), ps, st)
    
    x = ow_fold(X, size(y), W; stride=stride)
    d = ow_fold(D, size(y), W; stride=stride)
    return x ./ d, st
end

function Lux.initialstates(rng::AbstractRNG, n::CDLNet)
    st_l = Lux.initialstates(rng, n.lista)
    st_d = Lux.initialstates(rng, n.dictionary)
    return (lista=st_l, dictionary=st_d, csc=nothing)
end

function project!(l, ps, st, name) 
    if l isa Union{Lux.Conv, Lux.ConvTranspose}
        ps.weight ./= max.(1f0, sqrt.(sum(abs2, ps.weight, dims=(1,2))))

    elseif l isa GaborConvLayer
        ps.scale ./= max.(1f0, sqrt.(sum(abs2, kernel(l, ps, st), dims=(1,2))))

    elseif l isa Union{SoftThresh, GroupThresh}
        clamp!(ps.weight, 0f0, Inf)

    elseif l isa NonlocalSelfSim
        clamp!(ps.γ, 0f0, 1f0)
        if l.Wβ isa Lux.Conv
            clamp!(ps.Wβ.weight, 0f0, 1f0) 
        end

    elseif l isa LISTA
        for k=1:length(ps)
            project!(l.layer.analysis, ps[k].analysis, st.analysis, name*".layer_$k.analysis")
            project!(l.layer.synthesis, ps[k].synthesis, st.synthesis, name*".layer_$k.synthesis")
            project!(l.layer.prox, ps[k].prox, st.prox, name*".layer_$k.synthesis")
        end

    elseif l isa GroupLISTA
        for k=1:length(ps.layer)
            project!(l.layer.analysis, ps.layer[k].analysis, st.layer.analysis, name*".layer_$k.analysis")
            project!(l.layer.synthesis, ps.layer[k].synthesis, st.layer.synthesis, name*".layer_$k.synthesis")
            project!(l.layer.prox, ps.layer[k].prox, st.layer.prox, name*".layer_$k.synthesis")
        end
        project!(l.nlss, ps.nlss, st.nlss, name*".nlss")
    end
    return l, ps, st
end

function spectral_setup(rng::AbstractRNG, n::CDLNet; device=Lux.cpu) 
    ps_lista, st_lista = spectral_setup(rng, n.lista; device=device)

    @assert n.lista isa Union{LISTA, GroupLISTA}
    if n.lista isa LISTA
        ps = (lista=ps_lista, dictionary=deepcopy(ps_lista[1].synthesis))
        st = (lista=st_lista, dictionary=deepcopy(st_lista.synthesis))
    elseif n.lista isa GroupLISTA
        ps = (lista=ps_lista, dictionary=deepcopy(ps_lista.layer[1].synthesis))
        st = (lista=st_lista, dictionary=deepcopy(st_lista.layer.synthesis))
    end

    return ps, st
end

