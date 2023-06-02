using GroupCDLtip, Lux, CUDA, Random, Zygote, MLUtils, Statistics, NNlib
using ArgParse, Accessors
using CUDA.CUSPARSE
using FileIO, YAML, TensorBoardLogger
using TimerOutputs
using BenchmarkTools
using Images
using Printf
CUDA.allowscalar(false)

loadconfig(fn::String) = YAML.load_file(fn; dicttype=Dict{Symbol,Any})
saveconfig(fn::String, args::Dict{Symbol,Any}) = YAML.write_file(fn, args)

function main(fn; 
        seed=rand(1:9999), 
        fitcontinue=false, 
        device = CUDA.functional() ? gpu : cpu, 
        fit=false, 
        warmup=true, 
        eval=false, 
        eval_awgn=missing, 
        eval_dsetpath=missing, 
        eval_noise_dir=missing,
        eval_save_image=false,
        eval_nlss_windowsize=missing,
        eval_overlapping_window=false,
        eval_overlapping_window_stride=10,
        eval_save_image_dir=missing,
        infer_path=missing,
        infer_noise_path=missing,
        infer_save_adjmat=false,
        infer_save_image=false,
        infer_awgn=missing,
    )

    @info "Using device=$(device == gpu ? "gpu" : "cpu")"
    config = loadconfig(fn)
    net = CDLNet(;config[:net]...)
    @show Lux.parameterlength(net.lista)

    rng = Random.default_rng()
    Random.seed!(rng, seed)
    @info "random seed set to $seed"

    start = 1
    if config[:ckpt] != nothing && isfile(config[:ckpt])
        ckpt = load(config[:ckpt])
        @info "Loading checkpoint $(config[:ckpt]), epoch=$(ckpt[:epoch])"
        ps = ckpt[:ps] |> device
        st = Lux.initialstates(rng, net) |> device
        if fitcontinue
            start = ckpt[:epoch] + 1
            config[:opt][:lr] = ckpt[:lr]
        end
    elseif :pretrain_config in keys(config) && config[:pretrain_config] != nothing && isfile(config[:pretrain_config])
        config0 = loadconfig(config[:pretrain_config])
        net0 = CDLNet(; config0[:net]...)
        ckpt = load(config0[:ckpt])
        @info "Loading pretrain checkpoint $(config[:pretrain_config]), epoch=$(ckpt[:epoch])"
        ps0 = ckpt[:ps] |> device

        ps, st = Lux.setup(rng, net) .|> device
        for k=1:config0[:net][:K]
            @reset ps.lista.layer[k] = ps0.lista[k]
        end
        @reset ps.dictionary  = ps0.dictionary
    else
        ps, st = spectral_setup(rng, net; device=device) .|> device
        Lux.@layer_map project! net ps st
    end

    loaders = get_dataloaders(; rng=rng, config[:data]...)

    # warmup
    if warmup
        @info "Starting train warmup..."
        st = Lux.trainmode(st)
        x = loaders.train.data[1] |> unsqueeze(dims=4) |> device
        @show size(x)
        A = awgn(Identity(), [0.1f0;;;;] |> device)
        y = A(x, AddNoise())
        lossfun = mseloss
        @time CUDA.@sync compute_loss(lossfun, x, (y, A), net, ps, st)
        @time begin
            (loss, xhat, st), back = pullback(p -> compute_loss(lossfun, x, (y, A), net, p, st), ps)
            back((one(loss), nothing, nothing))[1]
        end
        @info "Train warmup completed."

        @info "Starting inference warmup (on test set)..."
        st = Lux.testmode(st)
        x = loaders.test.data[1] |> unsqueeze(dims=4) |> device
        @show size(x)
        A = awgn(Identity(), reshape([0f0], 1,1,1,1) |> device)
        y = A(x, AddNoise())
        @time CUDA.@sync begin
            xhat, st = net((y, A), ps, st)
        end
        @info "Inference warmup completed."
    end

    if fit
        if fitcontinue
            logger = TBLogger(config[:fit][:logdir], tb_append)
        else
            logger = TBLogger(config[:fit][:logdir]*"_$seed", tb_increment)
        end
        logdir = logger.logdir
        config[:fit][:logdir] = logdir

        st_opt, sched = create_optimiser_scheduler(ps; config[:opt]...)
        config[:ckpt] = joinpath(logdir, "net.bson")
        saveconfig(joinpath(logdir, "config.yaml"), config)

        ps, st = fit!(loaders, net, ps, st, st_opt, sched; 
                      device=device, 
                      logger=logger, 
                      start=start, 
                      rng=rng, 
                      config[:fit]...)
    end

    if eval
        eval_dsetname = basename(eval_dsetpath)

        @info "Performing eval over $eval_dsetname"
        eval_loader = image_dataloader(
            eval_dsetpath, 
            cropsize=nothing, 
            grayscale=net.in_chs==1 ? true : false, 
            batchsize=1, 
            parallel=false,
            shuffle=false,
            rng = rng,
        )

        if !ismissing(eval_nlss_windowsize)
            @info "setting nonlocal windowsize to $eval_nlss_windowsize"
            @reset st.lista.nlss.windowsize = eval_nlss_windowsize
        end

        st = Lux.testmode(st)
        σ_range = eval_awgn isa Number ? (eval_awgn,) : eval_awgn
        avgpsnr = zeros(length(σ_range))
        avgssim = zeros(length(σ_range))

        # write results to log file
        eval_logfn = joinpath(config[:fit][:logdir], "eval.csv")
        write_header_flag = !isfile(eval_logfn)
        io = open(eval_logfn, "a")

        # only write column headings if new file
        write_header_flag && write(io, "dataset, noise-level, psnr, ssim, time\n")

        for (i, σ) in enumerate(σ_range)
            if eval_save_image && ismissing(eval_save_image_dir)
                save_image_dir = joinpath(config[:fit][:logdir], basename(eval_dsetpath)*"-$(Int(σ))")
                mkpath(save_image_dir)
            elseif eval_save_image
                save_image_dir = joinpath(config[:fit][:logdir], basename(eval_dsetpath)*"-"*eval_save_image_dir)
                mkpath(save_image_dir)
            else
                save_image_dir = missing
            end

            to = TimerOutput()
            _, _, metrics = passthrough!(eval_loader, net, ps, st; 
                                         awgn_range = σ,
                                         desc = "eval, σ=$(σ):",
                                         show_progress = true,
                                         device = device,
                                         awgn_maxval = 255f0,
                                         save_image = eval_save_image,
                                         save_image_dir = save_image_dir,
                                         noise_dir = eval_noise_dir,
                                         overlapping_window = eval_overlapping_window,
                                         overlapping_window_stride = eval_overlapping_window_stride,
                                         timer = to,
                                         rng = rng)

            avgpsnr[i] = mean(metrics[:psnr])
            avgssim[i] = mean(metrics[:ssim])
            println("σ=$σ: PSNR=$(avgpsnr[i]) dB, SSIM=$(avgssim[i])")
            avgtime = TimerOutputs.time(to["inference"]) / TimerOutputs.ncalls(to["inference"]) / 1e9
            write(io, "$eval_dsetname, $σ, $(avgpsnr[i]), $(avgssim[i]), $avgtime\n")
            print_timer(to)
        end
        close(io)
    end

    if !ismissing(infer_path)
        st = Lux.testmode(st)
        x  = load_tensor(infer_path, GroupCDLtip.TestTransform(), Gray) |> unsqueeze(dims=4) |> device

        if !ismissing(infer_noise_path)
            ν = load(infer_noise_path) .* (infer_awgn/25f0) |> device
            y = x + ν
        else
            y = x + (infer_awgn/255f0) .* randn_like(x)
        end
        
        A = awgn(Identity(), infer_awgn/255f0)
        xhat, stout = net((y, A), ps, st)

        psnr = -10*log10(mean(abs2, x-xhat))
        ssim = GroupCDLtip.ssim(xhat, x; crop=true)
        @show psnr ssim
        
        if infer_save_image
            image_name = basename(infer_path) |> splitext |> first
            save_image_dir = joinpath(config[:fit][:logdir], "infer-$(infer_awgn)")
            mkpath(save_image_dir)

            fn = joinpath(save_image_dir, "$(image_name)_xhat_$(@sprintf("%.2f_%.2f", psnr, 100*ssim)).png")
            save(fn, clamp.(xhat[:,:,:,1],0,1) |> cpu)

            fn = joinpath(save_image_dir, "$(image_name)_y.png")
            save(fn, clamp.(y[:,:,:,1],0,1) |> cpu)

            normalize = x-> begin
                a, b = extrema(x)
                a == b ? zero(x) : (x .- a) ./ (b - a)
            end
            nrow = sqrt(net.subbands) |> Int

            absz  = abs.(stout.csc) |> cpu 
            absZ  = mosaicview(normalize(absz), nrow=nrow)
            absZn = mosaicview(mapslices(normalize, absz, dims=(1,2)), nrow=nrow)
            save(joinpath(save_image_dir, "$(image_name)_absz.png"), absZ)
            save(joinpath(save_image_dir, "$(image_name)_absz_eachnorm.png"), absZn)

            # save ada-thresholds
            if net isa GroupCDLtip.GroupCDLNet
                ξ = stout.lista.layer.prox.ξ |> cpu 
                ε = stout.lista.layer.prox.ε |> cpu
                τ = ps.lista.layer[end].prox.weight |> cpu

                Ξ  = mosaicview(normalize(ξ), nrow=nrow)
                Ξn = mosaicview(mapslices(normalize, ξ, dims=(1,2)), nrow=nrow)
                E  = mosaicview(normalize(ε), nrow=nrow)
                En = mosaicview(mapslices(normalize, ε, dims=(1,2)), nrow=nrow)

                Emt  = mosaicview(normalize(ε .- τ), nrow=nrow)
                Emtn = mosaicview(mapslices(normalize, ε .- τ, dims=(1,2)), nrow=nrow)

                save(joinpath(save_image_dir, "$(image_name)_thresh_xi.png"), Ξ)
                save(joinpath(save_image_dir, "$(image_name)_thresh_xi_eachnorm.png"), Ξn)
                save(joinpath(save_image_dir, "$(image_name)_thresh_epsilon.png"), E)
                save(joinpath(save_image_dir, "$(image_name)_thresh_epsilon_eachnorm.png"), En)
            end

            fn = joinpath(save_image_dir, "$(image_name)_n.npz")
            ismissing(infer_noise_path) && save(fn, y .- x |> cpu)
        end

        if infer_save_adjmat
            Γ = stout.lista.nlss.Σ
            G = reshape(Γ.nzVal, :, Γ.dims[1]) 
            G = reshape(cpu(G), 35, 35, 512, 512)
            save(joinpath(save_image_dir, "$(image_name)_G.npz"), G)
        end
    end

    return net, ps, st, loaders
end

function parse_commandline()
    s = ArgParseSettings(autofix_names=true)
    @add_arg_table! s begin
        "--seed", "-s"
            help = "random seed"
            arg_type = Int
            default = 0
        "--fit", "-f"
            help = "fit/train"
            action = :store_true
        "--fitcontinue", "-c"
            help = "continue fit/training from checkpoint"
            action = :store_true
        "--eval", "-e"
            help = "evaluate model"
            action = :store_true
        "--eval-dsetpath"
            help = "eval dataset path"
            arg_type = String
            default = "dataset/CBSD68"
        "--warmup"
            help = "do warmup pass"
            action = :store_true
        "--eval-awgn"
            default = 5:5:50
        "config"
            help = "config file"
            required = true
            arg_type = String
    end
    return parse_args(s; as_symbols=true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    device = CUDA.functional() ? gpu : cpu
    display(args)
    main(args[:config]; 
         seed=args[:seed], 
         fit=args[:fit], 
         fitcontinue=args[:fitcontinue], 
         warmup=args[:warmup],
         eval=args[:eval],
         eval_dsetpath=args[:eval_dsetpath],
         eval_awgn=(eval∘Meta.parse)(args[:eval_awgn]),
         device=device)
end

