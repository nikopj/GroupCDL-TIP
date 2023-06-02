function group_thresh(z::AbstractArray{T}, τ, Σ, st::NamedTuple, nlss, ps_nlss, st_nlss) where {T}
    # Z, latent: ws^2, channels, B
    # τ, thresh: 1, 1, channels, B
    # Σ, distance: ws^2, ws^2, B, row-normalized
    zα, _ = nlss.Wα(z, ps_nlss.Wα, st_nlss.Wα)
    sz = size(zα)
    Zα = reshape(zα, sz[1]*sz[2], sz[3], :) 
    ξ = sqrt.(Σ ⊠ abs2.(Zα) .+ 1f-8)
    ξ = reshape(ξ, sz)
    ξ, _ = nlss.Wβ(ξ, ps_nlss.Wβ, st_nlss.Wβ)
    return @. z * relu($one(T) - τ / (relu(ξ) + 1f-8)), st
end

function group_thresh(z::AbstractArray{T}, τ, Σ::AbstractSparseMatrix, st::NamedTuple, nlss, ps_nlss, st_nlss) where {T}
    # Z, latent: N, channels, B
    # τ, thresh: 1, 1, channels, B
    # Σ, distance: N, N, B
    # batchsize must be 1 for sparse-array inference
    zα, _ = nlss.Wα(z, ps_nlss.Wα, st_nlss.Wα)
    sz = size(zα)
    Zα = reshape(zα, sz[1]*sz[2], sz[3]) 
    ξ = sqrt.(Σ * abs2.(Zα) .+ 1f-8)
    ξ = reshape(ξ, sz)
    ξ, _ = nlss.Wβ(ξ, ps_nlss.Wβ, st_nlss.Wβ)
    ε = @. abs(z) * τ / (relu(ξ) + 1f-8)
    @. z = z * relu($one(T) - τ / (relu(ξ) + 1f-8))
    return z, (ε=ε, ξ=ξ)
end

const GroupThresh = Threshold{typeof(group_thresh)}

function GroupThresh(degree::Int, channels::Int, t0::Float32)
    return GroupThresh((1, 1, channels, degree+1), t0, ()->(ε=missing, ξ=missing))
end

function (l::GroupThresh)((z, σ)::Tuple{T1, T2}, ps, st::NamedTuple) where {T1, T2}
    τ = tensor_evalpoly(σ, ps.weight) 
    return soft_thresh.(z, τ), st
end

function (l::GroupThresh)((z, Σ, σ)::Tuple{T1, T2, T3}, ps, st::NamedTuple, nlss, ps_nlss, st_nlss) where {T1, T2, T3}
    τ = tensor_evalpoly(σ, ps.weight) 
    return group_thresh(z, τ, Σ, st, nlss, ps_nlss, st_nlss)
end

const GroupLISTALayer = PGMLayer{C, Ct, <: GroupThresh} where {C, Ct}

function (l::GroupLISTALayer)((y, H)::Tuple{<:AbstractArray, <:AbstractOperator}, ps, st::NamedTuple) 
    Ar, st_a = l.analysis(H'(y), ps.analysis, st.analysis)
    z,  st_p = l.prox((Ar, H.noise_level), ps.prox, st.prox)
    return z, (analysis=st_a, synthesis=st.synthesis, prox=st_p)
end

function (l::GroupLISTALayer)((z, Σ, y, H)::Tuple{T1, T2, T1, <:AbstractOperator}, ps, st::NamedTuple, nlss, ps_nlss, st_nlss) where {T1, T2}
    Bz, st_s = l.synthesis(z, ps.synthesis, st.synthesis)
    Ar, st_a = l.analysis(gramian(H)(Bz) - H'(y), ps.analysis, st.analysis)
    z, st_p  = l.prox((z - Ar, Σ, H.noise_level), ps.prox, st.prox, nlss, ps_nlss, st_nlss)
    return z, (analysis=st_a, synthesis=st_s, prox=st_p)
end

struct GroupLISTA{K, T <: GroupLISTALayer, S} <: Lux.AbstractExplicitLayer
    layer::T
    nlss::S
    init_Δupdate::Int
end

function Lux.initialstates(rng::AbstractRNG, L::GroupLISTA)
    st_layer = Lux.initialstates(rng, L.layer)
    st_nlss  = Lux.initialstates(rng, L.nlss)
    return (layer=st_layer, nlss=st_nlss, Δupdate=L.init_Δupdate)
end

function Lux.initialparameters(rng::AbstractRNG, L::GroupLISTA{K}) where {K}
    names = ntuple(k->Symbol("layer_$k"), Val(K))
    params = ntuple(k->Lux.initialparameters(rng, L.layer), Val(K))
    layer = NamedTuple{names}(params)
    nlss = Lux.initialparameters(rng, L.nlss)
    return (layer=layer, nlss=nlss)
end

function spectral_setup(rng::AbstractRNG, L::GroupLISTA{K}; device=Lux.cpu) where {K}
    psl0, stl0 = spectral_setup(rng, L.layer; device=device)

    names = ntuple(k->Symbol("layer_$k"), Val(K))
    params = ntuple(k->deepcopy(psl0), Val(K))
    ps_layer = NamedTuple{names}(params)

    ps_nlss, st_nlss = Lux.setup(rng, L.nlss)

    ps = (layer=ps_layer, nlss=ps_nlss)
    st = (layer=stl0, nlss=st_nlss, Δupdate=L.init_Δupdate)
    return ps, st
end

function (L::GroupLISTA{K})((y, H)::Tuple, ps, st::NamedTuple) where {K}
    @reset st.nlss.Σ  = missing
    @reset st.nlss.Σk = missing
    z, stl = L.layer((y, H), ps.layer[1], st.layer)
    Σ, stn = L.nlss((z, H), ps.nlss, st.nlss)
    z, stl = L.layer((z, Σ, y, H), ps.layer[2], stl, L.nlss, ps.nlss, stn)
    for k in 3:K
        if k % st.Δupdate == 0
            Σ, stn = L.nlss((z, H), ps.nlss, stn)
        end
        z, stl = L.layer((z, Σ, y, H), ps.layer[k], stl, L.nlss, ps.nlss, stn)
    end
    return z, (layer=stl, nlss=stn, Δupdate=st.Δupdate)
end
