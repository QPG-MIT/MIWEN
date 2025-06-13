#!/usr/bin/env python3

import sys
print("Python path:", sys.executable)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
# import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

from physics_copy import PhysicalConstants, NoiseModel, DiodeMixing
from inverse_comb_withbias import InverseComb

###############################################################################
# 1) SHIFT‐ONLY BATCHNORM
###############################################################################
class ShiftOnlyBatchNorm1d(nn.BatchNorm1d):
    """
    Subclass of nn.BatchNorm1d that:
      - keeps 'bias' trainable,
      - forces 'weight' (scale) = 1.0 (not learnable).
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=True)
        # Force the scale (weight) to always be 1.0 and disable its gradient
        with torch.no_grad():
            self.weight.data.fill_(1.0)
        self.weight.requires_grad_(False)
        
class ScalarScaleLayerNorm(nn.Module):
    """
    LayerNorm with a single learnable scalar γ and per‑feature bias β.
    Forward:  y = γ * x̂ + β,   where x̂ is the ε‑regularised, zero‑mean,
              unit‑variance normalisation over the last `normalized_shape`
              dimensions.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        # store shape/eps for use in forward
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # 1‑parameter scale (scalar γ) and vector bias β
        self.gamma = nn.Parameter(torch.tensor(1.0))               # shape ()
        self.beta  = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = F.layer_norm(x, self.normalized_shape,
                             weight=None, bias=None, eps=self.eps)
        return self.gamma * x_hat + self.beta

###############################################################################
# 2) TRAINABLE SCALAR AMPLIFICATION
###############################################################################
class TrainableScalarAmplification(nn.Module):
    """
    Applies a trainable scalar amplification to the input. The scalar is
    independent of time and is learned during training.
    """
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scalar

###############################################################################
# 3) ACTIVATION ENCODING
###############################################################################
class ActivationEncoding(nn.Module):
    """
    Encodes an input (batch, input_dim) into a time-domain waveform
    of shape (batch, n_t) by simple linear expansion. Here we append a bias
    so that we encode [x, 1] into time.
    """

    def __init__(self, n_t, input_dim, multfactor=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_t = n_t
        # self.multfactor = nn.Parameter(torch.tensor(multfactor, dtype=torch.float32))

    def forward(self, x, biasmag):
        batch_size, in_dim = x.shape
        if in_dim != self.input_dim:
            raise RuntimeError(f"Expected input dim {self.input_dim}, got {in_dim}.")

        # Append a bias value (1) to each input vector
        bias = biasmag*torch.ones(batch_size, 1, device=x.device)
        x_with_bias = torch.cat([x, bias], dim=1)  # (batch, input_dim+1)

        # Upsample the vector (now input_dim+1) to the time-domain length n_t
        encoded = F.interpolate(
            x_with_bias.unsqueeze(1),  # shape (batch, 1, input_dim+1)
            size=(self.n_t,),
            mode='linear',
            align_corners=False
        ).squeeze(1)

        return encoded

###############################################################################
# 4) WEIGHT ENCODING + MIXING
###############################################################################
class WeightEncodingandMixing(nn.Module):
    """
    Encodes a weight matrix (learnable) into a time waveform and then
    applies a diode formula mixing with the time-domain signal.
    """
    def __init__(self, n_t, input_dim, output_dim, bandwidth):
        super().__init__()
        self.n_t = n_t
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bandwidth = bandwidth

        # Create weight matrix with an extra column for bias
        self.W = nn.Parameter(torch.empty(output_dim, input_dim + 1))
        nn.init.kaiming_normal_(self.W, a=0.)
        # Scale factor
        self.W.data = self.W.data * 0.5

        # Physics setup
        self.constants = PhysicalConstants()
        self.noise_model = NoiseModel(self.constants, bandwidth=self.bandwidth)
        self.mixing = DiodeMixing(self.constants, self.noise_model)

        # For convenience, a constant multiplier (if needed)
        self.unit_converter = 1.0

    def forward(self, x_time):
        device = x_time.device

        # Flatten W and interpolate to match n_t
        W_flat = self.W.view(-1).unsqueeze(0)  # shape (1, output_dim*(input_dim+1))
        original_len = W_flat.shape[1]

        if original_len != self.n_t:
            W_float = W_flat.unsqueeze(1)  # shape (1,1,orig_len)
            W_up = F.interpolate(
                W_float,
                size=(self.n_t,),
                mode='linear',
                align_corners=False
            ).squeeze(1)
        else:
            W_up = W_flat

        Wtime = W_up.to(device)
        Wtime_b = Wtime.expand_as(x_time)  # match (batch, n_t)
        mixed_out = self._diode_formula(x_time, Wtime_b)
        return mixed_out

    def _diode_formula(self, z, w):
        mixed = self.mixing.exact_mixing(z, w)
        return mixed * self.unit_converter

###############################################################################
# 5) TIME TO LOGITS
###############################################################################
class TimeToLogits(nn.Module):
    """
    Splits the time-domain waveform evenly into n_classes segments,
    takes the mean of each segment as the class logit.
    """
    def __init__(self, n_t, n_classes=10):
        super().__init__()
        self.n_t = n_t
        self.n_classes = n_classes

    def forward(self, x):
        batch_size, length = x.size()
        if length < self.n_classes:
            raise ValueError(f"Time length={length} < n_classes={self.n_classes}")

        # Evenly split the time dimension into n_classes segments
        split_sizes = [length // self.n_classes] * (self.n_classes - 1)
        split_sizes += [length // self.n_classes + (length % self.n_classes)]
        segments = torch.split(x, split_sizes, dim=1)

        # For each segment, take mean along time dimension
        logits = torch.cat([seg.mean(dim=1, keepdim=True) for seg in segments], dim=1)
        return logits

###############################################################################
# 6) DYNAMIC MULTI-LAYER MODEL (with SHIFT‐ONLY BN in each layer)
###############################################################################
class MultiLayerTimeModel(nn.Module):
    """
    Build an L-layer network, layer_dims = [n1, n2, ..., n(L+1)].

    For each layer i, we do:
      - (First layer only) ActivationEncoding -> WeightEncodingandMixing -> Amplification -> ShiftOnlyBatchNorm1d -> InverseComb
      - (Middle layers) WeightEncodingandMixing -> Amplification -> ShiftOnlyBatchNorm1d -> InverseComb
      - (Last layer) WeightEncodingandMixing -> Amplification -> ShiftOnlyBatchNorm1d -> TimeToLogits
    """
    def __init__(self, layer_dims, t_steps, bandwidth):
        """
        layer_dims: e.g. [196, 100, 100, 10]
        t_steps   : always set to input_dim * first_hidden_dim
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.t_steps = t_steps
        self.bandwidth = bandwidth

        self.blocks = nn.ModuleList()

        # 1) First layer
        first_in = layer_dims[0]
        first_out = layer_dims[1]
        self.actenc = ActivationEncoding(n_t=self.t_steps, input_dim=first_in)
        
        block_first = nn.Sequential(
            WeightEncodingandMixing(n_t=self.t_steps, input_dim=first_in, output_dim=first_out, bandwidth=self.bandwidth),
            # TrainableScalarAmplification(init_value=1.0),
            # ShiftOnlyBatchNorm1d(self.t_steps),   # SHIFT ONLY BN
            ScalarScaleLayerNorm(self.t_steps),
            InverseComb(in_time_dim=self.t_steps, out_vector_dim=first_out)
        )
        self.blocks.append(block_first)

        # 2) Middle layers (layer i from 1..(num_layers-2) if we have >=3 total layers)
        for i in range(1, self.num_layers - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            block_mid = nn.Sequential(
                WeightEncodingandMixing(n_t=self.t_steps, input_dim=in_dim, output_dim=out_dim, bandwidth=self.bandwidth),
                # TrainableScalarAmplification(init_value=1.0),
                # ShiftOnlyBatchNorm1d(self.t_steps),
                ScalarScaleLayerNorm(self.t_steps),
                InverseComb(in_time_dim=self.t_steps, out_vector_dim=out_dim)
            )
            self.blocks.append(block_mid)

        # 3) Final layer (no InverseComb at the end)
        if self.num_layers >= 1:
            final_in = layer_dims[-2]
            final_out = layer_dims[-1]
            block_final = nn.Sequential(
                WeightEncodingandMixing(n_t=self.t_steps, input_dim=final_in, output_dim=final_out, bandwidth=self.bandwidth),
                # TrainableScalarAmplification(init_value=1.0),
                # ShiftOnlyBatchNorm1d(self.t_steps),
                ScalarScaleLayerNorm(self.t_steps),
                TimeToLogits(n_t=self.t_steps, n_classes=final_out)
            )
            self.blocks.append(block_final)

    def forward(self, x, biasmag):
        out = self.actenc(x, biasmag)
        for block in self.blocks:
            out = block(out)
        return out

###############################################################################
# 7) DATA UTILS (Loading MNIST for each input_dim)
###############################################################################
def get_full_mnist_datasets():
    """
    Returns trainset and testset in original 28x28 shape (only ToTensor).
    We'll resize dynamically later.
    """
    base_transform = T.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)
    return trainset, testset

def build_dataloaders_for_inputdim(trainset, testset, input_dim, batch_size=64):
    """
    For a given input_dim, compute side = sqrt(input_dim).
    Then dynamically resize 1x28x28 -> 1x(side)x(side), then flatten -> input_dim.
    """
    side = int(math.sqrt(input_dim))
    if side * side != input_dim:
        raise ValueError(f"Input dim {input_dim} is not a perfect square!")

    # We'll convert the raw 1x28x28 Tensor to a PIL image, resize, then back to Tensor
    transform_dynamic = T.Compose([
        T.ToPILImage(),
        T.Resize((side, side)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    train_wrapped = MNISTTransformWrapper(trainset, transform_dynamic)
    test_wrapped  = MNISTTransformWrapper(testset, transform_dynamic)

    trainloader = torch.utils.data.DataLoader(
        train_wrapped, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        test_wrapped, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return trainloader, testloader

class MNISTTransformWrapper(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, transform):
        self.dataset = mnist_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # 'img' is a 1x28x28 Tensor
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

###############################################################################
# 8) TRAINING UTILS
###############################################################################
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100.0 * correct / total
