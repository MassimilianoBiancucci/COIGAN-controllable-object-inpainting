import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from COIGAN.modules.simple_stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from COIGAN.modules.simple_stylegan2.base_modules import (
    StyledConv,
    ConstantInput,
    PixelNorm,
    EqualLinear,
    ConvLayer,
    ToRGB,
    ResBlock
)

"""
Implementation of stylegan2 without the skip connection
used for the progressive growing.
Are used only layers with the mudulated convolution and the weighted noise injection.
"""


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
        Simplified stylegan2 Generator initialization

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

        self.log_size = int(math.log(size, 2)) - 1
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, upsample=True, blur_kernel=blur_kernel
        )

        # the module is named to_rgb but the number of out channels is specified with the out_channels argument
        self.output = ToRGB(self.channels[2**self.log_size], style_dim, upsample=False, out_channels=self.out_ch)

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
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

            in_channel = out_channel

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

        # if there is more than one style code, then we use the mixing trick
        # the snippet below adjusts the style codes to be used for each layer
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            # if the inject index is not provided, then we randomly choose one
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            # concatenating the style codes and arranging it in the required format
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # starting the forward pass
        #############################################################################

        # NOTE: here the contant layer need the latent vector only to know the batch size
        out = self.input(latent) # generating the first constant tensor
        out = self.conv1(out, latent[:, 0], noise=noise[0]) # forwoed into the first convolutional layer
        # out here is a tensor bx512x4x4

        i = 1
        for conv1, conv2, noise1, noise2 in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2]
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)

            i += 2

        # a to_rgb layer with a kernel size of 1 is used to convert the output of the
        # last convolutional layer into an image of the given number of channels
        image = self.output(out, latent[:, -1])

        if return_latents:
            return image, latent

        else:
            return image, None


class Discriminator(nn.Module):
    def __init__(
        self, 
        size, 
        channel_multiplier=2, 
        blur_kernel=[1, 3, 3, 1], 
        input_channels=3
    ):
        super().__init__()

        self.in_ch = input_channels

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

        convs = [ConvLayer(self.in_ch, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

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

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

