import torch
import torch.nn as nn
import torch.nn.functional as F




class FeatureDownScale(nn.Module):
    def __init__(self, down_scale, in_c): # 这里的归一化层需要添加
        super(FeatureDownScale, self).__init__()
        # 利用卷积，进行patch partition+linear embedding 即实现 4倍下采样，这里的 embed_dim 即输出维度，是根据论文中给出的进行填写的
        self.proj = nn.Conv2d(in_c, in_c*2, kernel_size=down_scale, stride=down_scale)
        self.norm = nn.LayerNorm(in_c*2)
        self.act = nn.GELU()

    def forward(self, x):
        # X 输入图片 [B, C, H, W]
        # B, C, H, W = x.shape
        # x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        # 下采样patch_size倍 -> [H , out_dim, H/2, W/2]
        x = self.norm(self.proj(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)

class FeatureMiddleScale(nn.Module):
    def __init__(self, down_scale, in_c): # 这里的归一化层需要添加
        super(FeatureMiddleScale, self).__init__()
        # 利用卷积，进行patch partition+linear embedding 即实现 4倍下采样，这里的 embed_dim 即输出维度，是根据论文中给出的进行填写的
        self.proj = nn.Conv2d(in_c, in_c, kernel_size=down_scale, stride=down_scale)
        self.norm = nn.LayerNorm(in_c)
        self.act = nn.GELU()

    def forward(self, x):
        # X 输入图片 [B, C, H, W]
        # B, C, H, W = x.shape
        # x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        # 下采样patch_size倍 -> [H , out_dim, H/2, W/2]
        x = self.norm(self.proj(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)

class FeatureUpScale(nn.Module):
    def __init__(self, up_scale, in_c):
        super(FeatureUpScale, self).__init__()
        self.proj = nn.ConvTranspose2d(in_c, in_c, kernel_size=up_scale, stride=up_scale)
        self.norm = LayerNorm(in_c, eps=1e-6, data_format="channels_first")
    def forward(self, x):
        x = self.norm(self.proj(x))
        return x


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MultiHeadPatchScore(nn.Module):
    """[1,1,c]->[1,c]@[c,c]->[1,c]*[h*w,c]->[h,w,c]"""
    def __init__(self,
                 in_channels: int,
                 out_channel: int,
                 num_heads: int = 4
                 ):
        super(MultiHeadPatchScore, self).__init__()
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.scale: float = num_heads ** -0.5
        # Init layers
        self.k_map = nn.Linear(in_features=in_channels, out_features=out_channel, bias=True)
        self.score_map = nn.Linear(in_features=1, out_features=out_channel//num_heads, bias=True)
        self.k_v = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        # [B,C,H,W]->[B,C,1,1]
        # [B,C,1,1]->[B,C,1]
        # x_temp: [B,C,1]
        x_temp = x.mean((2, 3), keepdim=True).reshape(_B, _C, -1)
        # score_map: [B,C,C/head]
        # reshape:[B,num_head,c//num_head,C/head]
        x_se = self.score_map(x_temp).reshape(_B, self.num_heads, _C//self.num_heads, -1)

        # x_temp:[B,C,1] -> permute X_K [B,1,C]
        # x_k: [B,1,C]
        x_k = x_temp.permute(0, 2, 1)
        # reshape:[B,1,num_head,embed_dim_per_head]
        # permute:[B,num_head,1,embed_dim_per_head]
        x_k = self.k_map(x_k).reshape(_B, 1, self.num_heads, -1).permute(0, 2, 1, 3)

        # score:[B,num_head,1,C/head]
        # transpose:[B,num_head,C/head,1]
        # reshape:[B,C,1,1]
        score = self.softmax(x_k @ x_se.transpose(-2, -1) * self.scale).transpose(1, 2).reshape(_B, -1, 1, 1)
        # x[B,C,H,W]
        # score[B,C,1,1]
        x = self.proj(self.k_v(x) * score)
        # x[B,C,H,W]
        return x


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"Droprate={self.drop_prob}"

class Mlp_guorun(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    '''
        MPL 即 将输入节点在第一个linear后扩大四倍，在第二个linear后恢复原输入
    '''

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.norm = LayerNorm(in_features, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.pos = nn.Conv2d(hidden_features, hidden_features, 7, padding='same', groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.act = nn.GELU()
        self.skip = nn.Conv2d(in_features, out_features, 1)

        layer_scale_init_value = 1e-6
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((out_features)), requires_grad=True)
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        skip = self.skip(x)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        skip = skip + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x)
        return skip



class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.):
        """
        in_channels == out_channels

        :param in_channels:
        :param out_channels:
        :param num_heads:
        :param drop_path_rate:
        """
        super(DWBlock, self).__init__()

        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.scoreattn = MultiHeadPatchScore(in_channels=in_channels, out_channel=out_channels, num_heads=num_heads)

        self.mlp = Mlp_guorun(out_channels, out_channels*4, out_channels, drop_path_rate)


    def forward(self, x):
        # B1_shortcut [B,inc,H,W]
        shortcut = x

        x_attn = self.scoreattn(x)
        x_attn = x_attn + self.skip_path(shortcut)
        x_mlp = self.mlp(x_attn)

        return x_mlp


class DWBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.):
        """
        in_channels == out_channels

        :param in_channels:
        :param out_channels:
        :param num_heads:
        :param drop_path_rate:
        """
        super(DWBlockDecoder, self).__init__()

        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.scoreattn = MultiHeadPatchScore(in_channels=out_channels, out_channel=out_channels, num_heads=num_heads)

        self.mlp = Mlp_guorun(in_channels, out_channels*4, out_channels, drop_path_rate)


    def forward(self, x):
        # B1_shortcut [B,inc,H,W]

        shortcut = x_mlp = self.mlp(x)
        x_attn = self.scoreattn(x_mlp)
        x_attn = x_attn + self.skip_path(shortcut)

        return x_attn

class ScoreDown(nn.Module):
    """
    in_channels: input feature map channels
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, num_heads: int = 4, drop_path_rate=0.):
        super(ScoreDown, self).__init__()
        self.down = FeatureDownScale(down_scale=2, in_c=in_channels)
        self.dwBlock = DWBlock(out_channels, out_channels, num_heads, drop_path_rate)

    def forward(self, x):
        # x: [B,C,H,W]
        # down: [B,C,H/2,W/2]
        # dwBlock: [B,2C,H/2,W/2]
        x = self.down(x)
        x = self.dwBlock(x)
        return x

class ScoreMiddle(nn.Module):
    """
    in_channels: input feature map channels
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, num_heads: int = 4, drop_path_rate=0.):
        super(ScoreMiddle, self).__init__()
        self.down = FeatureMiddleScale(down_scale=2, in_c=in_channels)
        self.dwBlock = DWBlock(in_channels, out_channels, num_heads, drop_path_rate)

    def forward(self, x):
        # x: [B,C,H,W]
        # down: [B,C,H/2,W/2]
        # dwBlock: [B,2C,H/2,W/2]
        x = self.down(x)
        x = self.dwBlock(x)
        return x

class ScoreUp(nn.Module):
    """
    in_channels: input feature map channels. after upload , we will cat two feature map .[B,2C,H,W]
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, num_heads: int = 4, drop_path_rate=0.):
        super(ScoreUp, self).__init__()
        self.up = FeatureUpScale(up_scale=2, in_c=in_channels)

        self.dwBlock = DWBlockDecoder(2*in_channels, out_channels, num_heads, drop_path_rate)

    def forward(self, x1, x2):
        """
        :param x1: input feature map
        :param x2: concat feature map
        :return: [B,2C,H,W]
        """
        # x: [B,C,H,W]
        # up: [B,C,2*H,2*W]
        # cat: [B,2*C,2*H,2*W]
        # dwBlock: [B,C,2*H,2*W]
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.dwBlock(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_c=3, embed_dim=64):
        super().__init__()

        self.in_chans = in_c
        self.embed_dim = embed_dim
        # 利用卷积，进行patch partition+linear embedding 即实现 4倍下采样，这里的 embed_dim 即输出维度，是根据论文中给出的进行填写的
        self.proj1 = nn.Conv2d(in_c, embed_dim, kernel_size=1, padding='same', bias=False)

        self.norm1 = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        # X 输入图片 [B, C, H, W]
        B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act(x)
        return x


class ScoreOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ScoreOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SegConvFormer(nn.Module):
    def __init__(self, n_classes, embeddingdim = 64, num_heads: int = 4, drop_path_rate=0.25):
        super(SegConvFormer, self).__init__()
        self.n_classes = n_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 5)]


        self.embedding = PatchEmbedding(3, embeddingdim)          # 64,224,224
        self.inc = DWBlock(embeddingdim, embeddingdim, num_heads, dpr[0])   # 64,224,224
        self.down1 = ScoreDown(embeddingdim, embeddingdim*2, num_heads, dpr[1])  # in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.
        self.down2 = ScoreDown(embeddingdim*2, embeddingdim*4, num_heads, dpr[2])
        self.down3 = ScoreDown(embeddingdim*4, embeddingdim*8, num_heads, dpr[3])
        self.middle = ScoreMiddle(embeddingdim*8, embeddingdim*8, num_heads, dpr[4])
        self.up1 = ScoreUp(embeddingdim*8, embeddingdim*4, num_heads, dpr[3])  # in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.
        self.up2 = ScoreUp(embeddingdim*4, embeddingdim*2, num_heads, dpr[2])
        self.up3 = ScoreUp(embeddingdim*2, embeddingdim, num_heads, dpr[1])
        self.up4 = ScoreUp(embeddingdim, embeddingdim, num_heads, dpr[1])
        self.outc = ScoreOut(embeddingdim, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 填充0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        x = self.embedding(x)
        x1 = self.inc(x)#hw
        x2 = self.down1(x1)#h/2
        x3 = self.down2(x2)#h/4
        x4 = self.down3(x3)#h/8
        x5 = self.middle(x4)#h/8
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




if __name__ == '__main__':
    # from ptflops import get_model_complexity_info
    # import time
    #
    # num_classes = 4  # road=2, roadside=3
    #
    # divisor = 2
    #
    # # down = 2
    # # h = int(1080 / down) // 16 * 16
    # # w = int(1920 / down) // 16 * 16
    # h = 512
    # w = 512
    # print(h, w)
    #
    # net = ScoreNet(n_classes=4, num_heads=32, drop_path_rate=0.3).cuda()
    # image = (3, h, w)
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # print(f, p)
    #
    # s = time.clock()
    # with torch.no_grad():
    #     out = net(torch.randn(1, 3, h, w).cuda())
    # print(1 / (time.clock() - s))
    #
    # print(out.shape)

    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    print(torch.__version__)
    net = SegConvFormer(n_classes=4, embeddingdim=64, num_heads=1, drop_path_rate=0).cuda()
    # net = swin_base_patch4_window12_384().cuda()
    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 224, 224).cuda()
    # time_step=torch.tensor([999] * 1, device="cuda")
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # f, p = profile(net, inputs=(image, time_step))

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)