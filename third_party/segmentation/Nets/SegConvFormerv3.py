import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDownScale(nn.Module):
    def __init__(self, down_scale, in_c): # 这里的归一化层需要添加
        super(FeatureDownScale, self).__init__()
        # 利用卷积，进行patch partition+linear embedding 即实现 4倍下采样，这里的 embed_dim 即输出维度，是根据论文中给出的进行填写的
        self.proj = nn.Conv2d(in_c, in_c, kernel_size=down_scale, stride=down_scale)
        self.norm = nn.LayerNorm(in_c)

    def forward(self, x):
        # X 输入图片 [B, C, H, W]
        # B, C, H, W = x.shape
        # x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        # 下采样patch_size倍 -> [H , out_dim, H/2, W/2]
        x = self.norm(self.proj(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class FeatureUpScale(nn.Module):
    def __init__(self, up_scale, in_c):
        super(FeatureUpScale, self).__init__()
        self.proj = nn.ConvTranspose2d(in_c, in_c, kernel_size=up_scale, stride=up_scale)
        self.norm = nn.LayerNorm(in_c)
    def forward(self, x):
        x = self.norm(self.proj(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class GradModule(nn.Module):
    """[1,1,c]->[1,c]@[c,c]->[1,c]*[h*w,c]->[h,w,c]"""
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 32
                 ):
        super(GradModule, self).__init__()
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.scale: float = num_heads ** -0.5
        # Init layers
        self.k_map = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.score_map = nn.Linear(in_features=1, out_features=in_channels//num_heads, bias=True)
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
        x = x * score
        # x[B,C,H,W]
        return x

class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.):
        super(DWBlock, self).__init__()

        self.Fconv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.Fconv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.Fconv_11 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')

        self.DWconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same',groups=out_channels)
        # self.DWextern = nn.Linear(out_channels, 4*out_channels)
        # self.DWreverce = nn.Linear(4*out_channels, out_channels)
        self.DWextern = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1, padding='same')
        self.DWreverce = nn.Conv2d(4*out_channels, out_channels, kernel_size=1, padding='same')

        self.gelu = nn.GELU()  # 在DWextern后 Fconv_3后
        self.layernorm = nn.LayerNorm(out_channels)  # 在DWconv后
        self.batchnorm = nn.BatchNorm2d(out_channels)  # 在Fconv_11后

        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.score = GradModule(out_channels, num_heads)
        # 防止过拟合
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        # B1_shortcut [B,inc,H,W]
        B1_shortcut = x
        # x [B,outc,H,W]
        x = self.Fconv_11(self.gelu(self.batchnorm(self.Fconv_3(self.Fconv_1(x)))))
        x = self.drop_path(x) + self.skip_path(B1_shortcut)
        # B2_shortcut  [B,outc,H,W]
        B2_shortcut = x
        x = self.DWreverce(self.gelu(self.DWextern( self.layernorm( self.DWconv(x).permute(0, 2, 3, 1) ).permute(0, 3, 1, 2) )))
        x = self.drop_path(self.score(x)) + B2_shortcut
        return x

class ScoreDown(nn.Module):
    """
    in_channels: input feature map channels
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.):
        super(ScoreDown, self).__init__()
        self.down = FeatureDownScale(down_scale=2, in_c=in_channels)
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
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.):
        super(ScoreUp, self).__init__()
        self.up = FeatureUpScale(up_scale=2, in_c=in_channels)
        self.dwBlock = DWBlock(2*in_channels, out_channels, num_heads, drop_path_rate)

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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_path_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, drop_path_rate=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, drop_path_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class ScoreOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ScoreOut, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SegConvFormerV3(nn.Module):
    def __init__(self, n_classes, num_heads: int = 32, bilinear=True ,drop_path_rate=0.1):
        super(SegConvFormerV3, self).__init__()
        self.n_classes = n_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]


        self.inc = DWBlock(3, 64, num_heads, drop_path_rate)
        self.down1 = ScoreDown(64, 128, num_heads, dpr[0])  # in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.
        self.down2 = ScoreDown(128, 256, num_heads, dpr[1])
        self.down3 = ScoreDown(256, 512, num_heads, dpr[2])
        self.down4 = ScoreDown(512, 512, num_heads, dpr[3])
        self.up1 = Up(1024, 256, bilinear, dpr[0])  # in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.
        self.up2 = Up(512, 128, bilinear, dpr[1])
        self.up3 = Up(256, 64, bilinear, dpr[2])
        self.up4 = Up(128, 64, bilinear, dpr[3])
        self.outc = ScoreOut(64, n_classes)
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
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
    net = SegConvFormerV3(n_classes=4, num_heads=4, drop_path_rate=0.3).cuda()
    # net = swin_base_patch4_window12_384().cuda()
    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 512, 512).cuda()
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