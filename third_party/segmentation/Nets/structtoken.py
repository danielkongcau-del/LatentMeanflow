# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from torch.nn.init import trunc_normal_

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch.nn.functional as F

from functools import partial

import torch.nn as nn
from mmcv.cnn.bricks.drop import build_dropout


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int, optional): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(nn.Module):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = nn.LayerNorm(sample_dim)
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x = self.attn(self.norm1(x), identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(VisionTransformer, self).__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)
        if pretrained is not None:
            self.init_weights()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = torch.load(self.init_cfg['checkpoint'], map_location='cpu')

            print(f'Load pretrained model from {self.init_cfg["checkpoint"]}')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            msg=self.load_state_dict(state_dict, False)
            print(f'{msg}')
        elif self.init_cfg is not None:
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m, mode='fan_in')
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    nn.init.constant_(m, val=1.0)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoders head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

def nchw2nlc2nchw(module, x):
    """Flatten [N, C, H, W] shape tensor `x` to [N, L, C] shape tensor. Use the
    reshaped tensor as the input of `module`, and the convert the output of
    `module`, whose shape is.
    [N, L, C], to [N, C, H, W].
    Args:
        module: (Callable): A callable object the takes a tensor
            with shape [N, L, C] as input.
        x: (Tensor): The input tensor of shape [N, C, H, W].
    Returns:
        Tensor: The output tensor of shape [N, C, H, W].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> norm = nn.LayerNorm(4)
        >>> feature_map = torch.rand(4, 4, 5, 5)
        >>> output = nchw2nlc2nchw(norm, feature_map)
    """
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = module(x)
    x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
    return x


def nlc2nchw2nlc(module, x, hw_shape):
    """Convert [N, L, C] shape tensor `x` to [N, C, H, W] shape tensor. Use the
    reshaped tensor as the input of `module`, and convert the output of
    `module`, whose shape is.
    [N, C, H, W], to [N, L, C].
    Args:
        module: (Callable): A callable object the takes a tensor
            with shape [N, C, H, W] as input.
        x: (Tensor): The input tensor of shape [N, L, C].
        hw_shape: (Sequence[int]): The height and width of the
            feature map with shape [N, C, H, W].
    Returns:
        Tensor: The output tensor of shape [N, L, C].
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> conv = nn.Conv2d(16, 16, 3, 1, 1)
        >>> feature_map = torch.rand(4, 25, 16)
        >>> output = nlc2nchw2nlc(conv, feature_map, (5, 5))
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    x = x.transpose(1, 2).reshape(B, C, H, W)
    x = module(x)
    x = x.flatten(2).transpose(1, 2)
    return x

class ResBlock(nn.Module):
    def __init__(self, in_channels=19, channels=19):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               3,
                               1,
                               1)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               3,
                               1,
                               1)
        if channels != in_channels:
            self.identity_map = nn.Conv2d(in_channels,
                                          channels, 1, 1, 0)
        else:
            self.identity_map = nn.Identity()

    def forward(self, x):
        # refer to paper
        # Identity Mapping in Deep Residual Networks
        out = nchw2nlc2nchw(self.norm1, x)
        out = self.act1(out)
        out = self.conv1(out)
        out = nchw2nlc2nchw(self.norm2, out)
        out = self.act2(out)
        out = self.conv2(out)
        out = out + self.identity_map(x)

        return out


class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseConvBlock, self).__init__()
        mid_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels,
                               mid_channels,
                               3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(mid_channels,
                               out_channels,
                               1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = nchw2nlc2nchw(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = nchw2nlc2nchw(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        x = nchw2nlc2nchw(self.norm3, x)
        return x


class GroupConvBlock(nn.Module):
    def __init__(self,
                 embed_dims=150,
                 expand_ratio=6,
                 norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
                 dropout_layer=None,
                 init_cfg=None):
        super(GroupConvBlock, self).__init__()
        self.pwconv1 = nn.Conv2d(embed_dims,
                                 embed_dims * expand_ratio,
                                 1, 1)
        self.norm1 = build_norm_layer(norm_cfg,
                                      embed_dims * expand_ratio)[1]
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(embed_dims * expand_ratio,
                                embed_dims * expand_ratio,
                                3, 1, 1, groups=embed_dims)
        self.norm2 = build_norm_layer(norm_cfg,
                                      embed_dims * expand_ratio)[1]
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv2d(embed_dims * expand_ratio,
                                 embed_dims,
                                 1, 1)
        self.norm3 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.final_act = nn.GELU()
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        input = x
        x = self.pwconv1(x)
        x = nchw2nlc2nchw(self.norm1, x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = nchw2nlc2nchw(self.norm2, x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = nchw2nlc2nchw(self.norm3, x)

        if identity is None:
            x = input + self.dropout_layer(x)
        else:
            x = identity + self.dropout_layer(x)

        x = self.final_act(x)

        return x


class AttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=768,
                 query_dim=150,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(AttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, query, key, value):
        """x: B, C, H, W"""
        identity = query
        qb, qc, qh, qw = query.shape
        query = self.query_map(query).flatten(2)
        key = self.key_map(key).flatten(2)
        value = self.value_map(value).flatten(2)

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw)
        x = self.out_project(x)
        return identity + self.dropout_layer(self.proj_drop(x))


class PWConvAttentionLayer(nn.Module):
    def __init__(self,
                 dim=150,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(PWConvAttentionLayer, self).__init__()
        self.query_map = DepthWiseConvBlock(dim, dim)
        self.out_project = DepthWiseConvBlock(dim, dim)
        self.attn_pw_conv = nn.Conv2d(dim, dim, 1, 1)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, query):
        """x: B, C, H, W"""
        identity = query
        query = self.query_map(query)
        x = self.attn_pw_conv(query)
        x = self.out_project(x)
        return identity + self.dropout_layer(self.proj_drop(x))


class CrossSliceExtractionBlock(nn.Module):
    def __init__(self,
                 feature_channels=768,
                 num_classes=150,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 ffn_feature_maps=True):
        super(CrossSliceExtractionBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = AttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate,
                                         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio,
                                       norm_cfg=norm_cfg,
                                       dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio,
                                   norm_cfg=norm_cfg,
                                   dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, kernels, feature_maps):
        kernels = self.cross_attn(query=kernels,
                                  key=feature_maps,
                                  value=feature_maps)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            feature_maps = self.ffn2(feature_maps, identity=feature_maps)

        return kernels, feature_maps



class PointWiseExtractionBlock(nn.Module):
    def __init__(self,
                 feature_channels=768,
                 num_classes=150,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 ffn_feature_maps=True):
        super(PointWiseExtractionBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.attn = PWConvAttentionLayer(dim=feature_channels + num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate,
                                         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.ffn1 = GroupConvBlock(embed_dims=feature_channels + num_classes,
                                   expand_ratio=expand_ratio,
                                   norm_cfg=norm_cfg,
                                   dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, kernels, feature_maps):
        concated = torch.cat((kernels, feature_maps), dim=1)
        concated = self.attn(concated)

        concated = self.ffn1(concated, identity=concated)
        kernels = concated[:, :kernels.shape[1]]
        feature_maps = concated[:, kernels.shape[1]:]

        return kernels, feature_maps


class SelfSliceExtractionBlock(nn.Module):
    def __init__(self,
                 feature_channels=768,
                 num_classes=150,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 ffn_feature_maps=True):
        super(SelfSliceExtractionBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = AttentionLayer(kv_dim=feature_channels + num_classes,
                                         query_dim=feature_channels + num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate,
                                         dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.ffn1 = GroupConvBlock(embed_dims=feature_channels + num_classes,
                                   expand_ratio=expand_ratio,
                                   norm_cfg=norm_cfg,
                                   dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

    def forward(self, kernels, feature_maps):
        concated = torch.cat((kernels, feature_maps), dim=1)
        concated = self.cross_attn(query=concated,
                                   key=concated,
                                   value=concated)

        concated = self.ffn1(concated, identity=concated)
        kernels = concated[:, :kernels.shape[1]]
        feature_maps = concated[:, kernels.shape[1]:]

        return kernels, feature_maps


class CrossSliceExtracStructTokenHead(nn.Module):
    def __init__(
            self,
            align_corners=False,
            in_index=-1,
            input_transform=None,
            in_channels=768,
            channels=768,
            num_classes=150,
            norm_cfg=dict(type='LN',requires_grad=False),
            image_h=512,
            image_w=512,
            h_stride=16,
            w_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            interpolate_mode='bicubic',
            act_cfg=dict(type='GELU'),
            **kwargs):
        super(CrossSliceExtracStructTokenHead, self).__init__()
        self.image_h=image_h
        self.image_w=image_w
        self._init_inputs(in_channels, in_index, input_transform)
        self.align_corners = align_corners
        self.channels = channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        # del self.conv_seg
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.interpolate_mode = interpolate_mode
        self.has_odd = self.H % 2 != 0 or self.W % 2 != 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            CrossSliceExtractionBlock(feature_channels=self.channels,
                                      num_classes=self.num_classes,
                                      expand_ratio=mlp_ratio,
                                      drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate,
                                      drop_path_rate=dpr[i],
                                      ffn_feature_maps=i != num_layers - 1,
                                      norm_cfg=self.norm_cfg) for i in range(num_layers)])
        self.dec_proj = nn.Conv2d(self.in_channels,
                                  self.channels,
                                  1, 1)

        self.kernels = nn.Parameter(
            torch.randn(1, self.num_classes, self.H, self.W))

        # may use a large depth wise conv in the future
        self.residual_block = ResBlock(self.num_classes, self.num_classes)

    def init_weights(self):
        trunc_normal_(self.kernels, std=0.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m, std=.02, bias=0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m, mode='fan_in')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m, val=1.0)
        for layer in self.layers:
            layer.init_weights()

    def cls_seg(self, feat):
        raise NotImplementedError

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoders.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        feature_maps = self._transform_inputs(inputs)
        b, c, h, w = feature_maps.shape
        feature_maps = self.dec_proj(feature_maps)
        if (h, w) != (self.H, self.W):
            kernels = F.interpolate(input=self.kernels,
                                    size=(h, w),
                                    mode=self.interpolate_mode,
                                    align_corners=self.has_odd)
        else:
            kernels = self.kernels
        kernels = kernels.expand(b, -1, -1, -1)
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps)
        out = self.residual_block(kernels)
        out = resize(
            input=out,
            size=(self.image_h, self.image_w),
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    

class PointWiseExtracStructTokenHead(nn.Module):
    def __init__(
            self,
            align_corners=False,
            in_index=-1,
            input_transform=None,
            in_channels=768,
            channels=768,
            num_classes=150,
            norm_cfg=dict(type='LN',requires_grad=False),
            image_h=512,
            image_w=512,
            h_stride=16,
            w_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            interpolate_mode='bicubic',
            act_cfg=dict(type='GELU'),
            **kwargs):
        super(PointWiseExtracStructTokenHead, self).__init__()
        self.image_h=image_h
        self.image_w=image_w
        self._init_inputs(in_channels, in_index, input_transform)
        self.align_corners = align_corners
        self.channels = channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        # del self.conv_seg
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.interpolate_mode = interpolate_mode
        self.has_odd = self.H % 2 != 0 or self.W % 2 != 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            PointWiseExtractionBlock(feature_channels=self.channels,
                                     num_classes=self.num_classes,
                                     expand_ratio=mlp_ratio,
                                     drop_rate=drop_rate,
                                     attn_drop_rate=attn_drop_rate,
                                     drop_path_rate=dpr[i],
                                     ffn_feature_maps=True,
                                     norm_cfg=self.norm_cfg) for i in range(num_layers)])
        self.dec_proj = nn.Conv2d(self.in_channels,
                                  self.channels,
                                  1, 1)

        self.kernels = nn.Parameter(
            torch.randn(1, self.num_classes, self.H, self.W))

        # may use a large depth wise conv in the future
        self.residual_block = ResBlock(self.num_classes, self.num_classes)

    def init_weights(self):
        trunc_normal_(self.kernels, std=0.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m, std=.02, bias=0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m, mode='fan_in')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m, val=1.0)
        for layer in self.layers:
            layer.init_weights()

    def cls_seg(self, feat):
        raise NotImplementedError

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoders.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


    def forward(self, inputs):
        feature_maps = self._transform_inputs(inputs)
        b, c, h, w = feature_maps.shape
        feature_maps = self.dec_proj(feature_maps)
        if (h, w) != (self.H, self.W):
            kernels = F.interpolate(input=self.kernels,
                                    size=(h, w),
                                    mode=self.interpolate_mode,
                                    align_corners=self.has_odd)
        else:
            kernels = self.kernels
        kernels = kernels.expand(b, -1, -1, -1)
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps)
        out = self.residual_block(kernels)
        out = resize(
            input=out,
            size=(self.image_h, self.image_w),
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    

class SelfSliceExtracStructTokenHead(nn.Module):
    def __init__(
            self,
            align_corners=False,
            in_index=-1,
            input_transform=None,
            in_channels=768,
            channels=768,
            num_classes=150,
            norm_cfg=dict(type='LN',requires_grad=False),
            image_h=512,
            image_w=512,
            h_stride=16,
            w_stride=16,
            num_layers=2,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            interpolate_mode='bicubic',
            act_cfg=dict(type='GELU'),
            **kwargs):
        super(SelfSliceExtracStructTokenHead, self).__init__()
        self.image_h=image_h
        self.image_w=image_w
        self._init_inputs(in_channels, in_index, input_transform)
        self.align_corners = align_corners
        self.channels = channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        # del self.conv_seg
        self.H = image_h // h_stride
        self.W = image_w // w_stride
        self.interpolate_mode = interpolate_mode
        self.has_odd = self.H % 2 != 0 or self.W % 2 != 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            SelfSliceExtractionBlock(feature_channels=self.channels,
                                     num_classes=self.num_classes,
                                     expand_ratio=mlp_ratio,
                                     drop_rate=drop_rate,
                                     attn_drop_rate=attn_drop_rate,
                                     drop_path_rate=dpr[i],
                                     ffn_feature_maps=True,
                                     norm_cfg=self.norm_cfg) for i in range(num_layers)])
        self.dec_proj = nn.Conv2d(self.in_channels,
                                  self.channels,
                                  1, 1)

        self.kernels = nn.Parameter(
            torch.randn(1, self.num_classes, self.H, self.W))

        # may use a large depth wise conv in the future
        self.residual_block = ResBlock(self.num_classes, self.num_classes)

    def init_weights(self):
        trunc_normal_(self.kernels, std=0.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m, std=.02, bias=0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m, mode='fan_in')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m, val=1.0)
        for layer in self.layers:
            layer.init_weights()

    def cls_seg(self, feat):
        raise NotImplementedError


    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoders.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs


    def forward(self, inputs):
        feature_maps = self._transform_inputs(inputs)
        b, c, h, w = feature_maps.shape
        feature_maps = self.dec_proj(feature_maps)
        if (h, w) != (self.H, self.W):
            kernels = F.interpolate(input=self.kernels,
                                    size=(h, w),
                                    mode=self.interpolate_mode,
                                    align_corners=self.has_odd)
        else:
            kernels = self.kernels
        kernels = kernels.expand(b, -1, -1, -1)
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps)
        out = self.residual_block(kernels)
        out = resize(
            input=out,
            size=(self.image_h, self.image_w),
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    

class structtoken(nn.Module):
    def __init__(self, img_size=(512,512), in_chans=3, n_classes=4, choice_head='cse',pretrain_weight=None ):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_channels=in_chans,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic',
            pretrained=pretrain_weight,
            with_cp=True,
        )
        if choice_head == 'cse':
            self.decoder = CrossSliceExtracStructTokenHead(
                in_channels=768,
                channels=768,
                num_classes=n_classes,
                num_layers=4,
                norm_cfg=dict(type='LN', requires_grad=True),
                dropout_ratio=0.0,
                align_corners=False,
            )
        elif choice_head == 'pwe':
            self.decoder = PointWiseExtracStructTokenHead(
                in_channels=768,
                channels=768,
                num_classes=n_classes,
                num_layers=4,
                norm_cfg=dict(type='LN', requires_grad=True),
                dropout_ratio=0.0,
                align_corners=False,
            )
        elif choice_head == 'sse':
            self.decoder = SelfSliceExtracStructTokenHead(
                in_channels=768,
                channels=768,
                num_classes=n_classes,
                num_layers=4,
                norm_cfg=dict(type='LN', requires_grad=True),
                dropout_ratio=0.0,
                align_corners=False,
            )
        # self.encoder.init_weights()
        # self.decoder.init_weights()
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    from thop import profile
    import time

    print(torch.__version__)
    pretrained_weight = "/mnt/f/1.中国农大/数据集/vcnu_pretrain_weight/pretrain/pretrain_weights/VIT/B_32_imagenet1k.pth"
    net = structtoken(img_size=512, in_chans=3, n_classes=4, choice_head='cse',pretrain_weight=None).cuda()
    print(net)
    image = torch.rand(1, 3, 512, 512).cuda()
    f, p = profile(net, inputs=(image,))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))
    s = time.time()
    with torch.no_grad():
        out = net(image, )
    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))
    print(out.shape)

    # # throughput
    # batch_size = 2
    # num_batches = 80
    # input_data = torch.randn(1, 3, 512, 512).cuda()
    # def process_image(image):
    #     return net(image)
    # net.eval()
    # start_time = time.time()
    # with torch.no_grad():
    #     for batch in range(num_batches):
    #         output = process_image(input_data)
    # end_time = time.time()
    # total_images_processed = batch_size * num_batches
    # throughput = total_images_processed / (end_time - start_time)
    # print(f"throughput: {throughput:.2f} Image/s")  