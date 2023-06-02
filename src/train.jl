function create_optimiser_scheduler(ps; lr=1f-3, γ=1f0, step=1) 
    opt   = Adam(Float32(lr))
    sched = Step(Float32(lr), Float32(γ), step)
    return Optimisers.setup(opt, ps), sched
end

function total_gradnorm(∇)
    total = 0f0
    sumabs2(x) = begin
        isnothing(x) && return x
        total += sum(abs2, x)
        return x
    end
    fmap(sumabs2, ∇)
    return sqrt(total)
end

function clip_gradnorm(∇, clipnorm)
    ∇norm = total_gradnorm(∇)
    if ∇norm <= clipnorm 
        return ∇
    end
    normalize(x) = begin
        isnothing(x) && return x
        return x * clipnorm / ∇norm
    end
    return fmap(normalize, ∇)
end

function compute_loss(lossfun, target, (input, operator), model, ps, st)
    output, st = model((input, operator), ps, st)
    loss = lossfun(operator, target, output)
    return loss, output, st
end
mseloss(A, x, y) = mean(abs2, x - y)

function passthrough!(loader::DataLoader, net, Θ, Φ; 
        st_opt      = missing, 
        awgn_range  = 25f0, 
        awgn_maxval = 255f0,
        desc        = "", 
        device      = Lux.cpu,
        logger      = missing,
        show_progress = true,
        log_indices   = nothing,
        clipgrad      = nothing,
        global_epoch  = 0,
        save_image = false,
        save_image_dir = missing,
        noise_dir = missing,
        noise_scaled_loss = false,
        rng = Random.GLOABL_RNG,
        overlapping_window = false,
        overlapping_window_stride = 10,
        timer::Union{TimerOutput, Missing} = TimerOutput(),
    )
    training = ismissing(st_opt) ? false : true
    N = length(loader)

    t = Vector{Float32}(undef, N)
    if training
        Φ = Lux.trainmode(Φ) 
        metrics = Dict(:loss=>copy(t)) 
    else
        if overlapping_window
            Φ = Lux.trainmode(Φ) 
        else
            Φ = Lux.testmode(Φ) 
        end
        metrics = Dict(:loss=>copy(t), :psnr=>copy(t), :ssim=>copy(t))
    end

    if isa(awgn_range, Number)
        awgn_range = (awgn_range, awgn_range)
    end
    awgn_range = awgn_range ./ awgn_maxval .|> Float32

    lossfun = mseloss
    P = meter.Progress(length(loader), desc=desc, showspeed=true, enabled=show_progress)

    for (i, x) ∈ enumerate(loader)
        x = x |> device
        σ = rand_range(rng, x, awgn_range)
        A = awgn(Identity(), σ)
        y = A(rng, x, AddNoise())

        if training
            (loss, xhat, Φ), back = pullback(Θ -> compute_loss(lossfun, x, (y, A), net, Θ, Φ), Θ)
            ∇ = back((one(loss), nothing, nothing))[1]

            if isnan(loss) # signal need for backtracking
                return missing
            end
            if !isnothing(clipgrad)
                ∇ = clip_gradnorm(∇, Float32(clipgrad))
            end

            st_opt, Θ = Optimisers.update(st_opt, Θ, ∇)
            ∇norm = total_gradnorm(∇)
            Lux.@layer_map project! net Θ Φ

            metrics[:loss][i] = loss
            showvalues = [(:loss, loss), (:∇norm, ∇norm)]
        else
            if !ismissing(noise_dir)
                ν = load(joinpath(noise_dir, "$(i)_n.npz")) |> device
                y = x + ν
            end

            if !ismissing(timer)
                if overlapping_window
                    @timeit timer "inference" CUDA.@sync xhat, _ = ow_forward(net, y, A, Θ, Φ; stride=overlapping_window_stride)
                else
                    @timeit timer "inference" CUDA.@sync xhat, _ = net((y, A), Θ, Φ)
                end
            else
                if overlapping_window
                    xhat, _ = ow_forward(net, y, A, Θ, Φ; stride=overlapping_window_stride)
                else
                    xhat, _ = net((y, A), Θ, Φ)
                end
            end
            loss = lossfun(A, x, xhat)

            if isnan(loss)
                @warn "NaN at index i=$i, testmode"
            end

            metrics[:loss][i] = loss
            metrics[:psnr][i] = -10*log10(mean(abs2, x-xhat))
            metrics[:ssim][i] = ssim(xhat, x; crop=true)
            showvalues = [(k, metrics[k][i]) for k in keys(metrics)]

            if save_image
                save(joinpath(save_image_dir, 
                              # "$(i)_xhat_$(@sprintf("%.2f_%.2f", metrics[:psnr][i], 100*metrics[:ssim][i])).png"), 
                              "$(i)_xhat.png"), 
                     clamp.(xhat[:,:,:,1],0,1) |> Lux.cpu)

                save(joinpath(save_image_dir, "$(i)_y.png"), clamp.(y[:,:,:,1],0,1) |> Lux.cpu)
                ismissing(noise_dir) && save(joinpath(save_image_dir, "$(i)_n.npz"), y .- x |> Lux.cpu)
            end

            if !ismissing(logger) && !isnothing(log_indices) && i in log_indices
                clamp!(xhat, 0, 1)
                clamp!(y, 0, 1)
                res = abs.(x-xhat)
                tensor = cat(y, xhat, x, res; dims=4)
                log_mosaic(logger, "val_idx_$i", tensor; ncol=2, npad=10, normalize=false, step=global_epoch)
            end
        end

        meter.next!(P; showvalues=showvalues)
    end

    if show_progress
        for k in keys(metrics)
            print("AVG-$k=$(mean(metrics[k])), ")
        end
        println()
    end

    return Θ, Φ, metrics
end

function log_dict(f::Function, logger, dict; prefix="", kws...)
    for (key, value) in dict
        log_value(logger, prefix*string(key), f(value); kws...)
    end
end
log_dict(logger, dict; kws...) = log_dict(identity, logger, dict; kws...)

function log_mosaic(logger, name, tensor; step=logger.global_step, normalize=true, nrow=missing, npad=1, kws...)
    tensor = tensor |> Lux.cpu
    if normalize
        a, b = extrema(tensor)
        tensor = (tensor .- a)./(b-a)
    end
    if ismissing(nrow)
        N = size(tensor, ndims(tensor))
        nrow = ceil(Int, sqrt(N))
    end
    mos = cat([mosaicview(tensor[:,:,i,:]; npad=npad, nrow=nrow, kws...) for i=1:size(tensor,3)]..., dims=3)
    if size(mos, 3) == 3
        log_image(logger, name, mos, HWC; step=step)
    else
        log_image(logger, name, mos[:,:,1], HW; step=step)
    end
    return nothing
end

function fit!(loaders::NamedTuple, net, Θ, Φ, st_opt, sched::S; 
        epochs = 1, 
        start  = 1, 
        Δval   = 1, 
        Δsave  = 5,
        Δlog   = 1,
        δ      = 5,
        awgn_range = 25f0,  
        logdir     = "models/CDLNet-test", 
        verbose    = true,
        num_log_imgs = 4,
        logger::Union{TBLogger, Missing} = missing,
        device = Lux.cpu,
        rng = Random.GLOBAL_RNG,
        kws...
    ) where {S <: ParameterSchedulers.AbstractSchedule}
    logdir = logger.logdir
    val_log_indices = rand(rng, 1:length(loaders.val), num_log_imgs)

    bestloss = Dict(:train=>Inf, :val=>Inf, :test=>Inf)
    backtrack_count = 0
    backtrack_flag = false

    !ismissing(logger) && log_mosaic(logger, "dictionary", kernel(net.dictionary, Θ.dictionary, Φ.dictionary); step=0)
    save(joinpath(logdir, "0.bson"), :ps=> Θ |> Lux.cpu, :lr=>sched(0), :epoch=>0)
    
    epoch = start
    global_epoch = 1
    while epoch <= epochs 
        st_opt = Optimisers.adjust(st_opt, sched(epoch))
        !ismissing(logger) && log_value(logger, "lr", sched(epoch); step=global_epoch)
        !ismissing(logger) && log_value(logger, "backtrack", backtrack_count, step=global_epoch)

        for phase in (:train, :val, :test)
            phase == :val  && epoch % Δval != 0          && continue
            phase == :test && epoch < start + epochs - 1 && continue

             pt_out = passthrough!(loaders[phase], net, Θ, Φ; 
                st_opt     = phase==:train ? st_opt : missing,
                awgn_range = phase==:train ? awgn_range : mean(awgn_range),
                desc       = "Epoch-$phase [$epoch]",
                show_progress = verbose,
                log_indices   = phase==:val ? val_log_indices : nothing,
                logger        = epoch % Δlog == 0 ? logger : missing,
                global_epoch  = global_epoch,
                device = device,
                rng = rng,
                kws...
            )

            if ismissing(pt_out)
                backtrack_flag = true
                avgloss = -1f0
            else
                Θ, Φ, metrics = pt_out
                avgloss = mean(metrics[:loss])
                !ismissing(logger) && log_dict(mean, logger, metrics; prefix="$phase/", step=global_epoch)
            end

            if !backtrack_flag && avgloss < bestloss[phase]
                bestloss[phase] = avgloss
            elseif backtrack_flag || avgloss > δ*bestloss[phase]
                @info "fit!: avgloss=$(avgloss), backtracking..."
                backtrack_count += 1
                ckpt  = load(joinpath(logdir, epoch > Δsave ? "net.bson" : "0.bson"))
                Θ     = ckpt[:ps] |> device
                epoch = ckpt[:epoch]
                sched = Step(sched.start * 8f-1, sched.decay, sched.step_sizes.x)
                backtrack_flag = false
                break
            end

        end

        epoch % Δsave == 0 && save(joinpath(logdir, "net.bson"), :ps=> Θ |> Lux.cpu, :lr=>sched(epoch), :epoch=>epoch)
        epoch % Δlog  == 0 && !ismissing(logger) && log_mosaic(logger, "dictionary", kernel(net.dictionary, Θ.dictionary, Φ.dictionary); step=global_epoch)

        epoch += 1
        global_epoch += 1
    end

    return Θ, Φ
end

