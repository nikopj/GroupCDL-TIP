module GroupCDLtip

import Lux
using Adapt
using CUDA
using CUDA.CUSPARSE
using Random, NNlib
using Optimisers, Zygote
using ParameterSchedulers
using LinearAlgebra, Statistics, LazyGrids
using Accessors, Functors
import ChainRulesCore as CRC
using SparseArrays
using .Threads

using TimerOutputs, BenchmarkTools
using MLDatasets
using MLUtils
using Images
using DataAugmentation 
import DataAugmentation: Image, FromRandom, Maybe, Rotate, compose, ImageToTensor, apply
using Distributions: DiscreteNonParametric

using FileIO, Printf, BSON
using Logging, TensorBoardLogger, MosaicViews
import ProgressMeter as meter

include("solvers.jl")

include("operators.jl")
export awgn, Identity, AddNoise

include("adj/circulant.jl")
include("adj/softmax.jl")
include("adj/similarity.jl")
include("adj/adjacency.jl")

include("gabor.jl")
include("layers.jl")
include("nlss.jl")
include("group.jl")
include("networks.jl")
export CDLNet, create_optimiser_scheduler, spectral_setup, compute_loss, project!, gaborkernel

include("ssim.jl")
export ssim

include("data.jl")
export load_tensor, image_dataloader, get_dataloaders, ImageDataset, load_tensor

include("train.jl")
export fit!, passthrough!, σmseloss, mseloss

end
