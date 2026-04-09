import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# -------------------- 基础卷积模块 --------------------
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """ Depthwise Convolution """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups is None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=True)

    def forward(self, x):
        return self.depthwise(x)


# -------------------- U 形卷积编码器（按 512 输入） --------------------
class UEncoder(nn.Module):
    def __init__(self):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(3, 32)   # 64 -> 32
        self.pool1 = nn.MaxPool2d(2)

        self.res2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.res3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.res4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.res5 = DoubleConv(256, 512)
        self.pool5 = nn.MaxPool2d(2)


    def forward(self, x):
        features = []

        x = self.res1(x)          # 512 -> 512
        features.append(x)        # [0] (B,64,512,512)
        x = self.pool1(x)         # 256

        x = self.res2(x)          # 256
        features.append(x)        # [1] (B,128,256,256)
        x = self.pool2(x)         # 128

        x = self.res3(x)          # 128
        features.append(x)        # [2] (B,256,128,128)
        x = self.pool3(x)         # 64

        x = self.res4(x)          # 64
        features.append(x)        # [3] (B,512,64,64)
        x = self.pool4(x)         # 32

        x = self.res5(x)          # 32
        features.append(x)        # [4] (B,1024,32,32)
        x = self.pool5(x)         # 16

        features.append(x)        # [5] (B,1024,16,16)
        return features


# -------------------- 注意力 & Transformer 小块 --------------------
class Dual_axis(nn.Module):
    """
    这个里面原本就是按某个固定 img_size 去建 Bh/Bw 的，
    我们在 TransEncoder 里把 img_size 换成了 128/64/32/16，
    这里就会对应创建 128/64/32/16 的偏置。
    """
    def __init__(self, input_size, channels, d_h, d_v, d_w, heads, dropout):
        super(Dual_axis, self).__init__()
        self.dwconv_qh = DWCONV(channels, channels)
        self.dwconv_kh = DWCONV(channels, channels)
        # 这里原代码用了 (None, 1)，你原来能跑我就不改动语义了
        self.pool_qh = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_kh = nn.AdaptiveAvgPool2d((None, 1))
        self.fc_qh = nn.Linear(channels, heads * d_h)
        self.fc_kh = nn.Linear(channels, heads * d_h)

        self.dwconv_v = DWCONV(channels, channels)
        self.fc_v = nn.Linear(channels, heads * d_v)

        self.dwconv_qw = DWCONV(channels, channels)
        self.dwconv_kw = DWCONV(channels, channels)
        self.pool_qw = nn.AdaptiveAvgPool2d((1, None))
        self.pool_kw = nn.AdaptiveAvgPool2d((1, None))
        self.fc_qw = nn.Linear(channels, heads * d_w)
        self.fc_kw = nn.Linear(channels, heads * d_w)

        self.fc_o = nn.Linear(heads * d_v, channels)

        self.channels = channels
        self.d_h = d_h
        self.d_v = d_v
        self.d_w = d_w
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor_h = self.d_h ** -0.5
        self.scaled_factor_w = self.d_w ** -0.5
        # 这里的 input_size 就是 TransEncoder 传进来的 128/64/32/16
        self.Bh = nn.Parameter(torch.Tensor(1, self.heads, input_size, input_size), requires_grad=True)
        self.Bw = nn.Parameter(torch.Tensor(1, self.heads, input_size, input_size), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # 高度方向注意力
        qh = self.dwconv_qh(x)
        qh = self.pool_qh(qh).squeeze(-1).permute(0, 2, 1)
        qh = self.fc_qh(qh)
        qh = qh.view(b, h, self.heads, self.d_h).permute(0, 2, 1, 3).contiguous()

        kh = self.dwconv_kh(x)
        kh = self.pool_kh(kh).squeeze(-1).permute(0, 2, 1)
        kh = self.fc_kh(kh)
        kh = kh.view(b, h, self.heads, self.d_h).permute(0, 2, 1, 3).contiguous()

        attn_h = torch.einsum('... i d, ... j d -> ... i j', qh, kh) * self.scaled_factor_h
        attn_h = attn_h + self.Bh
        attn_h = torch.softmax(attn_h, dim=-1)

        # V
        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h, v_w, self.heads, self.d_v).permute(0, 3, 1, 2, 4).contiguous()
        v = v.view(v_b, self.heads, v_h, v_w * self.d_v).contiguous()

        # 宽度方向注意力
        qw = self.dwconv_qw(x)
        qw = self.pool_qw(qw).squeeze(-2).permute(0, 2, 1)
        qw = self.fc_qw(qw)
        qw = qw.view(b, w, self.heads, self.d_w).permute(0, 2, 1, 3).contiguous()

        kw = self.dwconv_kw(x)
        kw = self.pool_kw(kw).squeeze(-2).permute(0, 2, 1)
        kw = self.fc_kw(kw)
        kw = kw.view(b, w, self.heads, self.d_w).permute(0, 2, 1, 3).contiguous()

        attn_w = torch.einsum('... i d, ... j d -> ... i j', qw, kw) * self.scaled_factor_w
        attn_w = attn_w + self.Bw
        attn_w = torch.softmax(attn_w, dim=-1)

        # 双轴融合
        result = torch.matmul(attn_h, v)
        result = result.view(b, self.heads, h, w, self.d_v).permute(0, 1, 2, 4, 3).contiguous()
        result = result.view(b, self.heads, h * self.d_v, w).contiguous()
        result = torch.matmul(result, attn_w)
        result = result.view(b, self.heads, h, self.d_v, w).permute(0, 2, 4, 1, 3).contiguous()
        result = result.view(b, h * w, self.heads * self.d_v).contiguous()
        result = self.fc_o(result).view(b, self.channels, h, w)

        return result


class FFN_MultiLN(nn.Module):
    def __init__(self, in_channels, img_size, R, drop=0.):
        super(FFN_MultiLN, self).__init__()
        exp_channels = in_channels * R
        self.h = img_size
        self.w = img_size
        self.fc1 = nn.Linear(in_channels, exp_channels)
        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels)
        )
        self.ln1 = nn.LayerNorm(exp_channels, eps=1e-6)
        self.ln2 = nn.LayerNorm(exp_channels, eps=1e-6)
        self.ln3 = nn.LayerNorm(exp_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(exp_channels, in_channels)

    def forward(self, x):
        x = self.fc1(x)
        b, n, c = x.shape
        h = x

        x = x.view(b, self.h, self.w, c).permute(0, 3, 1, 2)
        x = self.dwconv(x).view(b, c, self.h * self.w).permute(0, 2, 1)
        x = self.ln1(x + h)
        x = self.ln2(x + h)
        x = self.ln3(x + h)
        x = self.act1(x)
        x = self.fc2(x)
        return x


class IntraTransBlock(nn.Module):
    def __init__(self, img_size, stride, d_h, d_v, d_w, num_heads, R=4, in_channels=46):
        super(IntraTransBlock, self).__init__()
        self.SlayerNorm = nn.LayerNorm(in_channels, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(in_channels, eps=1e-6)
        self.lmhsa = Dual_axis(img_size, in_channels, d_h, d_v, d_w, num_heads, 0.0)
        self.irffn = FFN_MultiLN(in_channels, img_size, R)

    def forward(self, x):
        x_pre = x
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.SlayerNorm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.lmhsa(x)
        x = x_pre + x

        x_pre = x
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.ElayerNorm(x)
        x = self.irffn(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = x_pre + x

        return x


# -------------------- 解码块 --------------------
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


# -------------------- Inter / Multi-scale Attention --------------------
class MLP(nn.Module):
    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape
        qkv = self.qkv_linear(x).reshape(
            B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head
        ).permute(4, 0, 1, 2, 5, 3, 6).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)
        return atten_value


class InterTransBlock(nn.Module):
    def __init__(self, dim):
        super(InterTransBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.FFN = MLP(dim)

    def forward(self, x):
        h = x
        x = self.SlayerNorm_1(x)
        x = self.Attention(x)
        x = h + x

        h = x
        x = self.SlayerNorm_2(x)
        x = self.FFN(x)
        x = h + x
        return x


class SpatialAwareTrans(nn.Module):
    def __init__(self, dim=128, num=1):   # 256 -> 128
        super(SpatialAwareTrans, self).__init__()
        self.ini_win_size = 2
        self.channels = [128, 256, 512, 512]  # 跟 TransEncoder 对上
        self.dim = dim
        self.depth = 4
        ...


        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        for i in range(self.depth):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))
        for i in range(self.depth):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = nn.Sequential(*[InterTransBlock(dim) for _ in range(self.num)])
        # 每一层拆成的 patch 数（不随输入图像大小，而随 win_size 规则）
        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]

    def forward(self, x):
        # x 是一个长度为 4 的 list，对应 128,64,32,16
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]
        # 做分块
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(
                0, 1, 3, 2, 4, 5
            ).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)

        for i in range(self.num):
            x = self.group_attention[i](x)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)

        # 掰回原来的空间
        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(
                0, 1, 3, 2, 4, 5
            ).contiguous().reshape(B, num_blocks * win_size, num_blocks * win_size, C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x


# -------------------- Transformer 编码器（改成 128,64,32,16） --------------------
class TransEncoder(nn.Module):
    def __init__(self):
        super(TransEncoder, self).__init__()
        # 512 输入 → UEncoder 输出是 [512,256,128,64,32,16]
        # 我们还是吃后四个：128,64,32,16
        self.block_layer = [1, 1, 1, 1]          # ← 每层 1 个 block
        self.size = [128, 64, 32, 16]
        self.channels = [128, 256, 512, 512]     # ← 通道整体减半

        self.R = 4

        # stage1：处理 128x128, 256ch
        stage1 = []
        for _ in range(self.block_layer[0]):
            stage1.append(
                IntraTransBlock(
                    img_size=self.size[0],
                    in_channels=self.channels[0],
                    stride=2,
                    d_h=self.channels[0] // 8,
                    d_v=self.channels[0] // 8,
                    d_w=self.channels[0] // 8,
                    num_heads=8,
                )
            )
        self.stage1 = nn.Sequential(*stage1)

        # stage2：处理 64x64, 512ch
        stage2 = []
        for _ in range(self.block_layer[1]):
            stage2.append(
                IntraTransBlock(
                    img_size=self.size[1],
                    in_channels=self.channels[1],
                    stride=2,
                    d_h=self.channels[1] // 8,
                    d_v=self.channels[1] // 8,
                    d_w=self.channels[1] // 8,
                    num_heads=8,
                )
            )
        self.stage2 = nn.Sequential(*stage2)

        # stage3：处理 32x32, 1024ch
        stage3 = []
        for _ in range(self.block_layer[2]):
            stage3.append(
                IntraTransBlock(
                    img_size=self.size[2],
                    in_channels=self.channels[2],
                    stride=2,
                    d_h=self.channels[2] // 8,
                    d_v=self.channels[2] // 8,
                    d_w=self.channels[2] // 8,
                    num_heads=8,
                )
            )
        self.stage3 = nn.Sequential(*stage3)

        # stage4：处理 16x16, 1024ch
        stage4 = []
        for _ in range(self.block_layer[3]):
            stage4.append(
                IntraTransBlock(
                    img_size=self.size[3],
                    in_channels=self.channels[3],
                    stride=1,
                    d_h=self.channels[3] // 8,
                    d_v=self.channels[3] // 8,
                    d_w=self.channels[3] // 8,
                    num_heads=8,
                )
            )
        self.stage4 = nn.Sequential(*stage4)

        # 下采样把上一层送给下一层拼接
        self.downlayers = nn.ModuleList()
        for i in range(len(self.block_layer) - 1):
            self.downlayers.append(
                ConvBNReLU(self.channels[i], self.channels[i] * 2, 2, 2, padding=0)
            )

        self.squeelayers = nn.ModuleList()
        for i in range(len(self.block_layer) - 2):
            self.squeelayers.append(
                nn.Conv2d(self.channels[i] * 4, self.channels[i] * 2, 1, 1)
            )
        self.squeeze_final = nn.Conv2d(self.channels[-1] * 3, self.channels[-1], 1, 1)

    def forward(self, x):
        # x 是 UEncoder 的 features: [512,256,128,64,32,16]
        _, _, feature0, feature1, feature2, feature3 = x  # 对应 128,64,32,16

        # 128x128, 256c
        feature0_trans = self.stage1(feature0)
        feature0_trans_down = self.downlayers[0](feature0_trans)  # -> 64x64, 512c

        # 64x64, 拼 UEncoder 的 64x64
        feature1_in = torch.cat((feature1, feature0_trans_down), dim=1)  # (512+512) = 1024
        feature1_in = self.squeelayers[0](feature1_in)                  # -> 512
        feature1_trans = self.stage2(feature1_in)
        feature1_trans_down = self.downlayers[1](feature1_trans)        # -> 32x32, 1024

        # 32x32
        feature2_in = torch.cat((feature2, feature1_trans_down), dim=1)  # (1024+1024)=2048
        feature2_in = self.squeelayers[1](feature2_in)                  # -> 1024
        feature2_trans = self.stage3(feature2_in)
        feature2_trans_down = self.downlayers[2](feature2_trans)        # -> 16x16, 2048

        # 16x16
        feature3_in = torch.cat((feature3, feature2_trans_down), dim=1) # (1024+2048)=3072
        feature3_in = self.squeeze_final(feature3_in)                   # -> 1024
        feature3_trans = self.stage4(feature3_in)

        return [feature0_trans, feature1_trans, feature2_trans, feature3_trans]


# -------------------- 并行编码器：卷积 + Transformer --------------------
class ParallEncoder(nn.Module):
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()
        self.Encoder2 = TransEncoder()
        self.num_module = 4
        self.fusion_list = [128, 256, 512, 512]   # ← 跟上面的改法一致
        self.inter_trans = SpatialAwareTrans(dim=128)

        self.squeelayers = nn.ModuleList()
        for i in range(self.num_module):
            self.squeelayers.append(
                nn.Conv2d(self.fusion_list[i] * 2, self.fusion_list[i], 1, 1)
            )

    def forward(self, x):
        skips = []
        features = self.Encoder1(x)          # 6 个尺度
        feature_trans = self.Encoder2(features)   # 4 个尺度: 128,64,32,16
        feature_trans = self.inter_trans(feature_trans)

        # 先把最前面两个高分辨率的卷积特征放进去 (512,256)
        skips.extend(features[:2])

        # 再把后面四个尺度做融合（transformer + 卷积）
        for i in range(self.num_module):
            # feature_trans[i]: 128,64,32,16
            # features[i+2]:    128,64,32,16
            skip = self.squeelayers[i](torch.cat((feature_trans[i], features[i + 2]), dim=1))
            skips.append(skip)
        return skips
        # 返回顺序：
        # [0] 512x512, 64
        # [1] 256x256, 128
        # [2] 128x128, 256
        # [3] 64x64,  512
        # [4] 32x32,  1024
        # [5] 16x16,  1024


# -------------------- 总模型 --------------------
class ScaleFormer(nn.Module):
    def __init__(self, num_classes):
        super(ScaleFormer, self).__init__()
        self.p_encoder = ParallEncoder()
        # skips: [512x512, 256x256, 128x128, 64x64, 32x32, 16x16]
        # channels:   32,       64,        128,     256,    512,    512
        self.encoder_channels = [512, 512, 256, 128, 64, 32]  # 我们自己标注一下也行

        # 16->32，接 32x32(512)
        self.decoder1 = DecoderBlock(512 + 512, 512)
        # 32->64，接 64x64(256)
        self.decoder2 = DecoderBlock(512 + 256, 256)
        # 64->128，接 128x128(128)
        self.decoder3 = DecoderBlock(256 + 128, 128)
        # 128->256，接 256x256(64)
        self.decoder4 = DecoderBlock(128 + 64, 64)
        # 256->512，接 512x512(32)
        self.decoder_final = DecoderBlock(64 + 32, 32)

        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=num_classes,
            kernel_size=3,
        )


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder_skips = self.p_encoder(x)
        # encoder_skips: [512x512, 256x256, 128x128, 64x64, 32x32, 16x16]

        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])  # 16 -> 32
        x2_up = self.decoder2(x1_up,        encoder_skips[-3])      # 32 -> 64
        x3_up = self.decoder3(x2_up,        encoder_skips[-4])      # 64 -> 128
        x4_up = self.decoder4(x3_up,        encoder_skips[-5])      # 128 -> 256

        # 这里会 up 到 512，然后跟最开始的 512x512 拼
        x_final = self.decoder_final(x4_up, encoder_skips[0])

        logits = self.segmentation_head(x_final)
        return logits



if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)

    net = ScaleFormer(num_classes=2, ).cuda()
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
