import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import torchinfo
from torchinfo import summary

"""
    缺点，维度太小，可引入盗梦空间的方法
    更改了位置信息所在位置！
"""
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

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    x = x.view(B, int(H // window_size), window_size, int(W // window_size), window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]    (H//Mh)*(W//Mw)正好为划分后的window的个数
    # permute 重塑矩阵后需要使用 contiguous 将其内存设为连续！
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # windows [B*num_windows, Mh, Mw, C]
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    # window shape [B*num_windows, Mh, Mw, C]
    # B : batch
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class ChannelTrans(nn.Module):
    def __init__(self, patch_size1=4, in_c=3, out_dim=48, norm_layer=None):
        super(ChannelTrans, self).__init__()
        self.patch_size = patch_size1
        self.in_chans = in_c
        self.out_dim = out_dim
        # 利用卷积，进行patch partition+linear embedding 即实现 4倍下采样，这里的 embed_dim 即输出维度，是根据论文中给出的进行填写的
        self.proj = nn.Conv2d(in_c, out_dim, kernel_size=patch_size1, stride=patch_size1)
        self.norm = norm_layer(out_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # X 输入图片 [B, C, H, W]
        _, _, H, W = x.shape
        # # padding
        # # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        # pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        # if pad_input:
        #     # to pad the last 3 dimensions,
        #     # (W_left, W_right, H_top,H_bottom, C_front, C_back)
        #     x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
        #                   0, self.patch_size[0] - H % self.patch_size[0],
        #                   0, 0))

        # 下采样patch_size倍 -> [H , out_dim, H/4, W/4]
        x = self.proj(x)
        _, _, H, W = x.shape
        # # flatten: [B, C, H, W] -> [B, C, HW]  即矩阵中 dim=1 维度上的每个元素的深度都为 HW .即一个patch,将1个patch矩阵[h,w]展平
        # # transpose: [B, C, HW] -> [B, HW, C]
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # X:[H, out_dim, H / 4, W / 4] H:H/4 , W: W/4
        return x
        #return x, H, W

class FeatureDownload(nn.Module):
    def __init__(self, patch_size, in_c, out_dim, norm_layer=None): # 这里的归一化层需要添加
        super(FeatureDownload, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_c
        self.out_dim = out_dim
        # 利用卷积，进行patch partition+linear embedding 即实现 4倍下采样，这里的 embed_dim 即输出维度，是根据论文中给出的进行填写的
        self.proj = nn.Conv2d(in_c, out_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(out_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # X 输入图片 [B, C, H, W]
        _, _, H, W = x.shape
        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍 -> [H , out_dim, H/4, W/4]
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        # X:[H, out_dim, H / 4, W / 4] H:H/4 , W: W/4
        return x



class WindowConvAttention(nn.Module):
    def __init__(self, dim, class_num, window_size, conv_drop=0., proj_drop=0.):
        super(WindowConvAttention, self).__init__()
        self.dim = dim
        self.class_num = class_num
        self.window_size = window_size

        # 引入绝对位置信息
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(dim, window_size[0], window_size[1]))  # [class_num, h ,w]

        # 窗口卷积，输入维度[B*num_windows, dim, H, W]-->[B*num_windows, dim*class_num, H, W]
        self.windowConv = nn.Conv2d(dim, class_num*dim, kernel_size=3, padding='same')
        # 通道注意力机制：[B*num_windows, Mh*Mw, dim*class_num] -> [B*num_windows, Mh*Mw, 1] 即通道融合后的feature map
        self.proj_channelAttention = nn.Linear(class_num*dim, 1)
        # 对融合后的feature map进行维度恢复，提取更加细腻的信息 [B*num_windows,1,Mh,Mw] -> [B*num_windows,class_num*dim,Mh,Mw]
        self.proj_channelExtend = nn.Conv2d(1, class_num*dim, kernel_size=1, padding='same')
        # 跳连接后，对融合后的特征信息进行维度恢复[B*num_windows,2*class_num*dim,Mh,Mw] -> [B*num_windows,dim,Mh,Mw]
        self.proj_merging = nn.Conv2d(class_num*dim, dim, kernel_size=3, padding='same')
        # 防止过拟合
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv_drop = nn.Dropout(conv_drop)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化位置编码表

    def forward(self, x):
        # [B*num_windows, dim, Mh, Mw]
        B_, C, H, W = x.shape
        x = x + self.relative_position_bias_table.unsqueeze(0)
        shortcut = x
        # windowConv: [B*num_windows, dim, Mh, Mw] -> [B*num_windows, dim*class_num, Mh, Mw]
        window_x = self.relu(self.windowConv(x))
        # # 这里的位置信息编码 还可以通过linear层，进行相乘！用作方案二 CON1*1实现矩阵对应位置相乘
        # # window_x:[B*num_windows, dim*class_num, Mh, Mw]
        # window_x = window_x + self.relative_position_bias_table.unsqueeze(0)  # [1, dim*class_num, h ,w]
        # shortcut = window_x

        # permute :  [B*num_windows, dim*class_num, Mh, Mw] ->  [B*num_windows, Mh, Mw, dim*class_num]
        # channelAttention: [B*num_windows, Mh, Mw, dim*class_num] -> [B*num_windows, Mh, Mw, 1]
        # permute : [B*num_windows, Mh, Mw, 1] -> [B*num_windows, 1, Mh, Mw]
        # channelExtend: [B*num_windows, 1, Mh, Mw] -> [B*num_windows, dim*class_num, Mh, Mw]
        window_x = window_x.permute(0, 2, 3, 1).contiguous()
        window_x = self.conv_drop(window_x)
        window_x = self.sigmoid(self.proj_channelAttention(window_x).permute(0, 3, 1, 2))
        window_x = self.relu(self.proj_channelExtend(window_x))
        # # 跳连接
        # #  [B*num_windows, dim*class_num, Mh, Mw] +  [B*num_windows, dim*class_num, Mh, Mw] =  [B*num_windows, dim*class_num, Mh, Mw]
        # window_x = window_x + shortcut

        #  [B*num_windows, dim*class_num, Mh, Mw] ->  [B*num_windows, dim, Mh, Mw]
        window_x = self.relu(self.proj_merging(window_x))
        window_x = window_x + shortcut
        window_x = self.proj_drop(window_x)
        # [B*num_windows, dim, Mh, Mw]
        return window_x

class GlobalConvAttention(nn.Module):
    def __init__(self, dim, class_num, window_size, window_number=1, conv_drop=0., proj_drop=0.):
        super(GlobalConvAttention, self).__init__()
        self.dim = dim
        self.class_num = class_num
        self.window_size = window_size

        self.proj_drop = nn.Dropout(proj_drop)
        self.conv_drop = nn.Dropout(conv_drop)
        # 全局卷积 [B, dim, Mh, Mw] -> [B, class_num*dim*window_number, Mh, Mw]
        self.conv = nn.Conv2d(dim, class_num*dim*window_number, kernel_size=3, padding='same', dilation=2)
        # 维度复原 [B, class_num*dim*window_number, Mh, Mw] -> [B, dim, Mh, Mw]
        self.recover = nn.Conv2d(class_num*dim*window_number, dim, kernel_size=3, padding='same', dilation=2)
        self.relu = nn.ReLU()

        self.channelatten = nn.Linear(dim, 1)
        self.featureatten = nn.Linear(window_size[0]*window_size[1], 1)


    def forward(self, x):
        # [B, dim, Mh, Mw]
        B_, C, H, W = x.shape

        shortcut = x
        # [B, dim, Mh, Mw] -> [B, class_num*dim*window_number, Mh, Mw]
        x = self.relu(self.conv(x))
        # [B, class_num*dim*window_number, Mh, Mw] -> [B, dim, Mh, Mw]
        x = self.relu(self.recover(x))
        # x: [B, dim, Mh, Mw]
        c_atten = x.permute(0, 2, 3, 1).contiguous()   # [B, Mh, Mw, dim]
        f_atten = x.view(B_, C, H*W)  # [B, dim, Mh*Mw]
        # channelatten : [B*num_windows, Mh, Mw, dim] -> [B*num_windows, Mh, Mw, 1]
        # view : [B*num_windows, Mh, Mw, 1] -> [B*num_windows, Mh*Mw, 1]
        # featureatten : [B, dim, Mh*Mw] -> [B*num_windows, dim, 1]
        # @, transpose: [B, Mh*Mw, dim]
        c_atten = self.channelatten(c_atten).view(B_, H*W, 1)
        f_atten = self.featureatten(f_atten)
        x = c_atten @ f_atten.transpose(1, 2)
        # x: [B*num_windows, Mh*Mw, dim] -> view:[B*num_windows, Mh, Mw, dim] -> permute[B*num_windows, dim, Mh, Mw]
        x = shortcut + x.view(B_, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.proj_drop(x)
        # [B, dim, Mh, Mw]
        return x

class WindowPositionConvBlock(nn.Module):
    """
    window_number = H*W/(window_size*window_size)
    """
    def __init__(self, in_dim, class_num, window_size, window_number, conv_drop=0., proj_drop=0., normal_layer=nn.BatchNorm2d):
        super(WindowPositionConvBlock, self).__init__()
        self.dim = in_dim
        self.class_num = class_num
        self.window_size = window_size
        self.window_number = window_number
        self.norm1 = normal_layer(in_dim)
        # # dim, class_num, window_size, conv_drop=0., proj_drop=0
        # self.atten = WindowConvAttention(
        #             in_dim, class_num, window_size, conv_drop, proj_drop)
        # #  dim, class_num, window_size, window_number, conv_drop=0., proj_drop=0
        # self.merg = GlobalConvAttention(
        #             in_dim, class_num, window_size, window_number, conv_drop, proj_drop)
        self.blocks = nn.ModuleList([
            WindowConvAttention(
                in_dim, class_num, window_size, conv_drop, proj_drop),
            GlobalConvAttention(
                in_dim, class_num, window_size, window_number, conv_drop, proj_drop)
        ])

        self.norm2 = normal_layer(in_dim)

    def forward(self, x):
        # [B, C, H, W ]
        B, C, H, W = x.shape
        # shortcut [B, C, H, W ]
        shortcut = x
        # permute后X:[B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = self.norm1(x)

        # # [B,H,W,C]
        # # pad feature maps to multiples of window size
        # # 把feature map给pad到window size的整数倍
        # pad_l = pad_t = 0
        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # window_partition:[B,H,W,C] -> (num_windows*B, window_size, window_size, C)
        # permute: (num_windows*B, window_size, window_size, C) -> [B*num_windows, dim, Mh, Mw]
        x_windows = window_partition(x, self.window_size).permute(0, 3, 1, 2).contiguous()
        # [B*num_windows, dim, Mh, Mw] -> [B*num_windows, dim, Mh, Mw]
        atten_window = self.blocks[0](x_windows)
        atten_window = atten_window + shortcut
        # shifted_x: (num_windows*B, window_size, window_size, C) -> (B, H, W, C)
        shifted_x = window_reverse(atten_window.permute(0, 2, 3, 1), self.window_size, Hp, Wp)
        shifted_x = self.norm1(shifted_x)
        # (B, H, W, C) -> [B, dim, Mh, Mw]
        atten_window = self.blocks[1](shifted_x.permute(0, 3, 1, 2).contiguous())
        x = self.norm2(atten_window)

        # [B, C, H, W]
        return atten_window

class WindowNet(nn.Module):
    """
    patch_size: 下采样倍数
    """
    def __init__(self,
                 in_HW=224,
                 patch_sizes=4,
                 in_chans=3,
                 extre_dim=48,
                 dim_c=96,
                 class_num=2,
                 window_size=7,
                 depths=(4, 4, 6, 2),
                 conv_drop=0.1,
                 proj_drop=0.1,
                 norml_layers=nn.BatchNorm2d
                 ):
        super(WindowNet, self).__init__()
        self.HW = in_HW
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, proj_drop, sum(depths))]
        self.layers = nn.ModuleList()
        # [224,224,3]->[56,56,48]
        layer = ChannelTrans(patch_size1=patch_sizes, in_c=in_chans, out_dim=extre_dim, norm_layer=norml_layers)  # patch_size=4(下采样倍数), in_c=3, out_dim=48, norm_layer=None
        self.layers.append(layer)
        for i_block in range(len(depths)):
            for i_layer in range(depths[i_block]):
                layer = WindowPositionConvBlock(
                    in_dim=extre_dim*2**i_block,
                    class_num=class_num,
                    window_size=(self.window_size, self.window_size),
                    window_number=int((in_HW/(4*2**i_block))**2/self.window_size**2),
                    conv_drop=conv_drop,
                    proj_drop=dpr[sum(depths[:i_layer])],
                    normal_layer=norml_layers
                )  # in_dim, class_num, window_size, **window_number, conv_drop=0., proj_drop=0, norm_layer=nn.BatchNorm
                self.layers.append(layer)
            if i_block != len(depths)-1:
                layer = FeatureDownload(patch_size=int(patch_sizes*2**i_block), in_c=extre_dim*2**i_block,
                                        out_dim=dim_c*2**i_block, norm_layer=norml_layers)  # patch_size, in_c, out_dim, norm_layer=None
                self.layers.append(layer)

        self.num_layers = len(depths)
        # stage4输出特征矩阵的channels
        self.num_features = int(dim_c * 2 ** (self.num_layers - 1))  # 8C
        self.norm = norml_layers(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.segconv1_outdim = int(dim_c * 2 ** (self.num_layers - 2))  # 4C
        self.segconv2_outdim = int(dim_c * 2 ** (self.num_layers - 3))  # 2C
        self.segconv3_outdim = int(dim_c * 2 ** (self.num_layers - 4))  # C
        self.segconv4_outdim = extre_dim  # 48
        # [H/32,W/32,8C] -> [H/16,W/16,4C]
        self.upconv1 = nn.ConvTranspose2d(self.num_features*2, self.segconv1_outdim, kernel_size=2, stride=2)
        # [H/16,W/16,4C] -> [H/16,W/16,4C]  该层*2，提取特征
        self.segconv1 = nn.Conv2d(self.segconv1_outdim, self.segconv1_outdim, kernel_size=3, padding='same')
        # self.segnorm1 = nn.LayerNorm()

        # [H/16,W/16,4C] -> [H/8,W/8,2C]
        self.upconv2 = nn.ConvTranspose2d(self.segconv1_outdim*2, self.segconv2_outdim, kernel_size=2, stride=2)
        # [H/8,W/8,2C] -> [H/8,W/8,2C]  该层*2，提取特征
        self.segconv2 = nn.Conv2d(self.segconv2_outdim, self.segconv2_outdim, kernel_size=3, padding='same')

        # [H/8,W/8,2C] -> [H/4,W/4,C]
        self.upconv3 = nn.ConvTranspose2d(self.segconv2_outdim*2, self.segconv3_outdim, kernel_size=2, stride=2)
        # [H/4,W/4,C]-> [H/4,W/4,C]  该层*2，提取特征
        self.segconv3 = nn.Conv2d(self.segconv3_outdim, self.segconv3_outdim, kernel_size=3, padding='same')

        # [H/4,W/4,C] -> [H/2,W/2,48]
        self.upconv4 = nn.ConvTranspose2d(self.segconv3_outdim*2, self.segconv4_outdim, kernel_size=2, stride=2)
        # [H/2,W/2,48] -> [H/2,W/2,48]  该层*2，提取特征
        self.segconv4 = nn.Conv2d(self.segconv4_outdim, self.segconv4_outdim, kernel_size=3, padding='same')

        # [H/2,W/2,48] -> [H,W,48]
        self.upconv5 = nn.ConvTranspose2d(self.segconv4_outdim, self.segconv4_outdim, kernel_size=2,stride=2)
        # [H,W,48] -> [H,W,class_num]  输出
        self.segconv5 = nn.Conv2d(self.segconv4_outdim, class_num, kernel_size=1, padding='same')

        # softmax [B,C,H,W]
        self.softmax = nn.Softmax(dim=-1)
        self.extre_drop = nn.Dropout(p=dpr[0])
        self.relu = nn.ReLU()


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 填充0
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        # x : [B,C,H,W]
        res = []
        for layer in self.layers:
            x = layer(x)
            res.append(x)
            if layer == 0:
                x = self.extre_drop(x)

        x = self.relu(self.upconv1(x))
        # x = F.layer_norm()
        x = self.relu(self.segconv1(x))
        x = self.relu(self.segconv1(x))
        x = self.extre_drop(x)

        x = torch.cat((x, res[7]), 1)
        x = self.relu(self.upconv2(x))
        x = self.relu(self.segconv2(x))
        x = self.relu(self.segconv2(x))
        x = self.extre_drop(x)

        x = torch.cat((x, res[3]), 1)
        x = self.relu(self.upconv3(x))
        x = self.relu(self.segconv3(x))
        x = self.relu(self.segconv3(x))
        x = self.extre_drop(x)

        x = torch.cat((x, res[1]), 1)
        x = self.relu(self.upconv4(x))
        x = self.relu(self.segconv4(x))
        x = self.relu(self.segconv4(x))
        x = self.extre_drop(x)

        x = self.relu(self.upconv5(x))
        x = self.softmax(self.segconv5(x))
        return x

def windowNet_window7_224(num_classes=2):
    model = WindowNet(in_HW=224,  # 输入图像宽高[224,224]
                      in_chans=3,  # 输入维度 3
                      patch_sizes=4,  # 第一次的缩放比例 H/4， W/4
                      extre_dim=48,  # 第一次扩维后的维度[H/4,W/4,extre_dim]
                      dim_c=96,  # stage的维度
                      class_num=num_classes,  # 分类个数
                      window_size=7,  # 窗口大小
                      depths=(1, 1, 3, 1),  #共有几个stage
                      norml_layers=nn.BatchNorm2d
                     )   # in_HW=224, patch_size=4, in_chans=3, extre_dim=48, dim_c=96, class_num=2,
                        # window_size=7, depths=(4, 4, 6, 2), conv_drop=0.1, proj_drop=0.1,
                        # norm_layer=nn.LayerNorm
    return model




"""-------------------------------------------------------------------------"""
class windtest(nn.Module):
    def __init__(self,
                 in_HW=224,
                 patch_sizes=4,
                 in_chans=3,
                 extre_dim=48,
                 dim_c=96,
                 class_num=2,
                 window_size=7,
                 depths=(4, 4, 6, 2),
                 conv_drop=0.1,
                 proj_drop=0.1,
                 norml_layers=nn.BatchNorm2d
                 ):
        super(windtest, self).__init__()
        self.HW = in_HW
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, proj_drop, sum(depths))]
        self.layers = nn.ModuleList()
        # [224,224,3]->[56,56,48]
        layer = ChannelTrans(patch_size1=patch_sizes, in_c=in_chans, out_dim=extre_dim, norm_layer=norml_layers)
        self.layers.append(layer)
        for i_layer in range(depths[0]):
            layer = WindowPositionConvBlock(
                in_dim=extre_dim,
                class_num=class_num,
                window_size=(self.window_size, self.window_size),
                window_number=int((in_HW/(4*2**0))**2/self.window_size**2),
                conv_drop=conv_drop,
                proj_drop=proj_drop,
                normal_layer=norml_layers
            )  # in_dim, class_num, window_size, **window_number, conv_drop=0., proj_drop=0, norm_layer=nn.BatchNorm
            self.layers.append(layer)
        layer = FeatureDownload(patch_size=int(patch_sizes * 2 ** 0), in_c=extre_dim*2**0,
                                out_dim=dim_c*2**0,
                                norm_layer=norml_layers)  # patch_size, in_c, out_dim, norm_layer=None
        self.layers.append(layer)

        for i_layer in range(depths[1]):
            layer = WindowPositionConvBlock(
                in_dim=extre_dim * 2 ** 1,
                class_num=class_num,
                window_size=(self.window_size, self.window_size),
                window_number=int((in_HW/(4*2**1))**2/self.window_size**2),
                conv_drop=conv_drop,
                proj_drop=proj_drop,
                normal_layer=norml_layers
            )  # in_dim, class_num, window_size, **window_number, conv_drop=0., proj_drop=0, norm_layer=nn.BatchNorm
            self.layers.append(layer)

        layer = FeatureDownload(patch_size=int(patch_sizes * 2 ** 1), in_c=extre_dim * 2 ** 1,
                                out_dim=dim_c * 2 ** 1,
                                norm_layer=norml_layers)  # patch_size, in_c, out_dim, norm_layer=None
        self.layers.append(layer)



        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 填充0
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        # x : [B,C,H,W]
        for layer in self.layers:
            x = layer(x)
            if layer == 0:
                x = self.extre_drop(x)

        return x




def test(numclass=2):
    model = windtest(
        in_HW=224,  # 输入图像宽高[224,224]
        in_chans=3,  # 输入维度 3
        patch_sizes=4,  # 第一次的缩放比例 H/4， W/4
        extre_dim=48,  # 第一次扩维后的维度[H/4,W/4,extre_dim]
        dim_c=96,  # stage的维度
        class_num=numclass,  # 分类个数
        window_size=7,  # 窗口大小
        depths=(1, 1, 3, 1),  # 共有几个stage
        norml_layers=nn.BatchNorm2d
    )
    return model

# model = windowNet_window7_224(numclass=4)
# print(model)
# model = test(numclass=4)
# print(summary(model))













