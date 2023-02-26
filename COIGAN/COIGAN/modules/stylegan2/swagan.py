import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from typing import Optional

from COIGAN.modules.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from COIGAN.modules.stylegan2.stylegan2 import (
    ModulatedConv2d,
    StyledConv,
    ConstantInput,
    PixelNorm,
    Upsample,
    Downsample,
    Blur,
    EqualLinear,
    ConvLayer,
)


def get_haar_wavelet():
    """
    Get Haar Wavelet
    """
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


def dwt_init(x):
    """
    Discrete Wavelet Transform initialization
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    """
    Inverse Discrete Wavelet Transform initialization
    """
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = (
        in_batch,
        int(in_channel / (r ** 2)),
        r * in_height,
        r * in_width,
    )
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel : out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2 : out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3 : out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class HaarTransform(nn.Module):
    """
    Module for Haar Transform
    """

    def __init__(self):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet()

        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):

    def __init__(self):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet()

        self.register_buffer("ll", ll)
        self.register_buffer("lh", -lh)
        self.register_buffer("hl", -hl)
        self.register_buffer("hh", hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh


class ToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], out_channels=3):
        """
        ToRGB layer
        Apply a modulated conv layer to the input and then apply a bias
        if the skip input is present from the last layer it 

        """
        super().__init__()

        self.out_ch = out_channels # channels of the final output tensor

        if upsample:
            self.iwt = InverseHaarTransform()
            self.upsample = Upsample(blur_kernel)
            self.dwt = HaarTransform()

        self.conv = ModulatedConv2d(in_channel, self.out_ch * 4, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, self.out_ch * 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)

            out = out + skip

        return out


class Generator(nn.Module):

    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        out_channels=3
    ):
        """
        SWAGAN Generator initialization

        Args:
            size (int): output tensor size
            style_dim (int): style vector dimension
            n_mlp (int): number of MLP layers
            channel_multiplier (int): channel multiplier
            blur_kernel (list): blur kernel
            lr_mlp (float): learning rate multiplier for the mapping network
            out_channels (int): number of output channels

        """
        super().__init__()

        self.size = size # h, w of output tensor

        self.style_dim = style_dim # style vector dimension

        self.out_ch = out_channels

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, out_channels=self.out_ch)

        self.log_size = int(math.log(size, 2)) - 1
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        # generating noise tensors for each layer
        # calculating create the noise tensors for each layer
        # and store it as a buffer in the noises module
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, 
                    out_channel, 
                    3, 
                    style_dim, 
                    blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, out_channels=self.out_ch))

            in_channel = out_channel

        self.iwt = InverseHaarTransform()

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        """
            Method to generate images from the generator.

            Args:
                styles (torch.Tensor): Style codes to be used for image generation.
                return_latents (bool): If True, returns the latents used for image generation.
                inject_index (int): If not None, injects the given style code at the given index.
                truncation (float): Truncation factor to be used for truncation trick.
                truncation_latent (torch.Tensor): Latent vector to be used for truncation trick.
                input_is_latent (bool): If True, assumes that the input is a latent vector.
                noise (list): List of noise tensors to be used for image generation.
                randomize_noise (bool): If True, randomizes noise for each image in the batch.
            
            Returns:
                torch.Tensor: Generated images.
        """
        ### input preprocessing
        #############################################################################

        # if the input is a latent vector, then we don't need to pass it through the style network
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        # if the noise isn't provided, then we create a new noise tensor if randomize_noise is True
        # else we use the noise tensor stored in the noises module
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        # if truncation is not 1, then we use the truncation trick
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        # if there is more than one style code, then use the mixing trick
        # the snippet below adjusts the style codes to be used for each layer
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            # if the inject index is not provided, then randomly choose one
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            # concatenating the style codes and arranging it in the required format
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # starting the forward pass
        #############################################################################

        out = self.input(latent) # generating the first constant tensor
        out = self.conv1(out, latent[:, 0], noise=noise[0]) # forwoed into the first convolutional layer
        # out here is a tensor bx512x4x4

        skip = self.to_rgb1(out, latent[:, 1]) # generating the first skip connection
        # skip here is a tensor bx12x8x8 (to_rgb double the size of the input and return the image in the wavelet domain)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = self.iwt(skip)

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class FromRGB(nn.Module):

    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1], in_channels=3):
        """
            Skip connection from the network output to the wavelet branch.

            Args:
                out_channel (int): Number of channels in the output.
                downsample (bool): If True, downsamples the input by a factor of 2.
                blur_kernel (list): Blur kernel to be used for downsampling.
                in_channels (int): Number of channels in the final tensor, if the image is rgb then it is 3.
        """
        super().__init__()

        self.downsample = downsample

        self.in_ch = in_channels

        if downsample:
            self.iwt = InverseHaarTransform()
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform()

        self.conv = ConvLayer(self.in_ch * 4, out_channel, 3)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):

    def __init__(
        self, 
        size, 
        channel_multiplier=2, 
        blur_kernel=[1, 3, 3, 1], 
        input_channels=3,
        keep_features: bool = False
    ):
        """
        SWAGAN discriminator.

        Args:
            size (int): input size.
            channel_multiplier (int): Multiplier for the number of channels of conv layers.
            blur_kernel (list): Blur kernel to be used for downsampling.
            input_channels (int): Number of channels in the input image.
            features (bool): If True, the inference return the intermediate features.
        """
        super().__init__()

        self.in_ch = input_channels
        self.keep_features = keep_features
        self.features = []

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.dwt = HaarTransform()

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, in_channels=self.in_ch))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4], in_channels=self.in_ch))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        input = self.dwt(input)
        out = None
        self.features = []

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)  
            self.features.append(out) # store the features
            
        _, out = self.from_rgbs[-1](input, out)
        self.features.append(out) # store the features

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)

        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)

        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        self.features.append(out) # store the features

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out, self.features

