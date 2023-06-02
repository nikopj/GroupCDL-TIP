using YAML
using Base.Iterators: product

loadconfig(fn::String) = YAML.load_file(fn; dicttype=Dict{Symbol,Any})
saveconfig(fn::String, args::Dict{Symbol,Any}) = YAML.write_file(fn, args)

function makeconfigs(loopd::Dict, fn::String="scripts/config.yaml"; name=missing, startnum=1, savedir="hpc/args.d", joint=false)
    based = loadconfig(fn)

    if ismissing(name)
        name = join(keys(loopd), "_")
    end

    summary_fn = joinpath(savedir, "summary_"*name*".txt")
    io = open(summary_fn, "a")

    newds = factory(based, loopd; io=io, joint=joint)
    for i = 1:length(newds)
        ver = string(startnum + i - 1)
        namei = name * "-" * ver
        newds[i][:fit][:logdir] = joinpath("models", namei)
        println(io, namei)
        println(namei)
        !ismissing(savedir) && saveconfig(joinpath(savedir, namei*".yaml"), newds[i])
    end

    close(io)
    return newds
end

function factory(based::Dict, loopd::Dict; io=stdout, joint=false)
    if joint
        l = length.(values(loopd))
        @assert all(l .== first(l)) "length of all loop vectors must match, got l=$l"
        iter = zip([loopd[k] for k in keys(loopd)]...)
    else
        iter = product([loopd[k] for k in keys(loopd)]...)
    end
    newdvec = []
    println(io, keys(loopd))
    io != stdout && println(keys(loopd))
    for t in iter
        println(io, t)
        io != stdout && println(t)
        newd = deepcopy(based)
        for (k, v) in zip(keys(loopd), t)
            setrecursive!(newd, k, v) && begin @warn "makeconfigs: did not find key $k"; return end
        end
        push!(newdvec, newd)
    end
    return newdvec
end

function setrecursive!(d::Dict, key, value)
    if key in keys(d)
        d[key] = value
        return false
    end
    flag = true
    for k in keys(d)
        if d[k] isa Dict
            flag = setrecursive!(d[k], key, value)
            !flag && break # end greedy search
        end
    end
    return flag
end

