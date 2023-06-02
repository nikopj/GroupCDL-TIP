struct Collect <: DataAugmentation.Transform end

function apply(::Collect, image::DataAugmentation.ArrayItem; randstate = nothing)
    return ArrayItem(collect(itemdata(image)))
end

TrainTransform(cs::Integer) = compose(
    Maybe(FlipX(), 0.5),
    Maybe(FlipY(), 0.5),
    Rotate(DiscreteNonParametric([0,90,180,270], 0.25*ones(4))),
    RandomCrop((cs, cs)), 
    ImageToTensor(), 
    Collect(),
)

TestTransform() = ImageToTensor()
load_tensor(path, tfm=ImageToTensor(), C=identity) = load(path) .|> C |> Image |> x->apply(tfm, x) |> itemdata 
LoadTensor(tfm, C=identity) = path -> load_tensor(path, tfm, C)

struct ImageDataset <: MLDatasets.AbstractDataContainer
    transform::DataAugmentation.Transform
    images::NTuple{N, Matrix} where {N}
    paths::Vector{String}
end

ImageDataset(paths) = ImageDataset(ImageToTensor(), load.(paths), paths)
ImageDataset(transform, ds::MLDatasets.FileDataset) = ImageDataset(transform, ntuple(i->ds[i], length(ds)), ds.paths)

function Base.getindex(ds::ImageDataset, i::Integer) 
    item = Image(ds.images[i])
    item = apply(ds.transform, item)
    return itemdata(item) 
end
Base.getindex(ds::ImageDataset, is::AbstractVector) = map(Base.Fix1(getobs, ds), is)
Base.length(ds::ImageDataset) = length(ds.images)

function image_dataloader(dirs::String...; cached=false, cropsize=nothing, grayscale=true, kws...)
    C = grayscale ? Gray : RGB
    ds = begin
        ds = FileDataset(x->load(x) .|> C, vcat([readdir(d; join=true) for d in dirs]...))
        if isnothing(cropsize) || ismissing(cropsize)
            transform = TestTransform()
        else
            transform = TrainTransform(cropsize)
        end
        ds = ImageDataset(transform, ds)
        cached ? CachedDataset(ds) : ds
    end
    return DataLoader(ds; collate=true, kws...)
end
image_dataloader(dirs::Vector{String}; kws...) = image_dataloader(dirs...; kws...)

# collate batches items into single array
# partial drops last mini-batch if not up to batchsize
function get_dataloaders(;
        trainpath = missing, 
        valpath   = missing, 
        testpath  = missing,
        cropsize  = 128, 
        batchsize = 10, 
        grayscale = true,
        parallel = true,
        buffer = false,
        cached = false,
        rng = Random.GLOBAL_RNG,
    )
    dl_train = image_dataloader(
        trainpath,
        cached    = cached,
        cropsize  = cropsize,
        grayscale = grayscale,
        batchsize = batchsize, 
        collate   = true, 
        partial   = false,
        buffer    = buffer,
        parallel  = parallel,
        shuffle   = true,
        rng = rng,
    )

    dl_val = image_dataloader(
        valpath,
        cached    = false,
        cropsize  = missing,
        grayscale = grayscale,
        batchsize = 1, 
        collate   = true, 
        partial   = true,
        buffer    = false,
        parallel  = false,
        shuffle   = false,
        rng = rng,
    )
    dl_test = image_dataloader(
        testpath,
        cached    = false,
        cropsize  = missing,
        grayscale = grayscale,
        batchsize = 1, 
        collate   = true, 
        partial   = true,
        buffer    = false,
        parallel  = false,
        shuffle   = false,
        rng = rng, 
    )
    return (train=dl_train, val=dl_val, test=dl_test)
end

