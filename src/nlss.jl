struct NonlocalSelfSim{W1, W2, W3, F, T} <: Lux.AbstractExplicitContainerLayer{(:Wθ, :Wϕ)}
    Wθ::W1
    Wϕ::W1
    Wα::W2
    Wβ::W3
    simfun::F
    init_γ::T
    init_windowsize::Int
end

function NonlocalSelfSim(M, simtype::String, ws::Int, γ0::Real=0.8f0; compressed=true)
    simtype = simtype |> lowercase
    if simtype == "distance"
        simfun = distance_similarity
    elseif simtype == "dot"
        simfun = dot_similarity
    else
        throw("Invalid argument: simtype=$simtype must be \"distance\" or \"dot\"")
    end
    return NonlocalSelfSim(M, simfun, ws, γ0; compressed=compressed)
end

function NonlocalSelfSim(map::Pair, simfun::Function, ws::Int, γ0::T; compressed=true) where {T}
    if compressed == false
        Wθ = Lux.NoOpLayer()
        Wϕ = Lux.NoOpLayer()
        Wα = Lux.NoOpLayer()
        Wβ = Lux.NoOpLayer()
    elseif compressed == "similarity"
        Wθ = Lux.Conv((1,1), map; bias=false)
        Wϕ = Lux.Conv((1,1), map; bias=false)
        Wα = Lux.NoOpLayer()
        Wβ = Lux.NoOpLayer()
    elseif compressed == "all" || compressed
        Wθ = Lux.Conv((1,1), map; bias=false)
        Wϕ = Lux.Conv((1,1), map; bias=false)
        Wα = Lux.Conv((1,1), map; bias=false)
        Wβ = Lux.Conv((1,1), reverse(map); bias=false)
    else
        throw("compressed=$compressed must be false, true, \"all\", or \"similarity\".")
    end
    return NonlocalSelfSim{typeof(Wθ), typeof(Wα), typeof(Wβ), typeof(simfun), typeof(γ0)}(Wθ, Wϕ, Wα, Wβ, simfun, γ0, ws)
end

function Lux.initialparameters(rng::AbstractRNG, l::NonlocalSelfSim)
    psθ = Lux.initialparameters(rng, l.Wθ)
    psϕ = deepcopy(psθ)
    psα = Lux.initialparameters(rng, l.Wα)
    psβ = Lux.initialparameters(rng, l.Wβ)
    if !(l.Wα isa Lux.NoOpLayer)
        W = copy(psθ.weight)
        psα.weight .= W
        #println("opnorm α", opnorm(psα.weight[1,1,:,:]))
    end
    if !(l.Wβ isa Lux.NoOpLayer)
        psβ.weight .= rand_like(psβ.weight) ./ 2sqrt(4 + sum(size(psβ.weight)))
        #println("opnorm β", opnorm(psβ.weight[1,1,:,:]))
    end
    return (Wθ=psθ, Wϕ=psϕ, Wα=psα, Wβ=psβ, γ=[l.init_γ])
end

function Lux.initialstates(rng::AbstractRNG, l::NonlocalSelfSim)
    stθ = Lux.initialstates(rng, l.Wθ)
    stϕ = Lux.initialstates(rng, l.Wϕ)
    stα = Lux.initialstates(rng, l.Wα)
    stβ = Lux.initialstates(rng, l.Wβ)
    return (Σ=missing, Σk=missing, Wθ=stθ, Wϕ=stϕ, Wα=stα, Wβ=stβ, windowsize=l.init_windowsize, training=Val(false))
end

function (l::NonlocalSelfSim)((x, H)::Tuple, ps, st::NamedTuple) 
    xθ, stθ = l.Wθ(x, ps.Wθ, st.Wθ)
    xϕ, stϕ = l.Wϕ(x, ps.Wϕ, st.Wϕ)
    Σ, Σk = adjacency(st.Σ, st.Σk, l.simfun, xθ, xϕ, st.windowsize, ps.γ, st.training)
    return Σ, (Σ=Σ, Σk=Σk, Wθ=stθ, Wϕ=stϕ, Wα=st.Wα, Wβ=st.Wβ, windowsize=st.windowsize, training=st.training) 
end

