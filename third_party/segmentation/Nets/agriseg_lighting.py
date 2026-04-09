import torch
import torch.nn as nn
import torch.nn.functional as F
from Nets.cooperative_games_moe import DifferenceMoE, DifferenceMoEWithRouter

from Nets.kv_attention import MultiKVAttention

from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from math import sqrt

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_channels=3, out_channels=64, patch_size=4, drop_rate=0.):
        super(Embeddings, self).__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()


    def forward(self, x):
        x = self.dropout(self.norm(self.patch_embeddings(x)))  
        return x

class InverseEmbeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_channels=64, out_channels=64, patch_size=4, bilinear=True, drop_path_rate=0.):
        super(InverseEmbeddings, self).__init__()
                # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)

        self.module = ExpertModule(out_channels,drop_path_rate)

    def forward(self, x):
        #x = self.dropout(self.norm(self.up(x)))  
        x = self.up(x)
        x = self.module(x)
        return x

class ExpertModule(nn.Module):
    def __init__(self,out_channels,drop_path_rate=0.):
        super().__init__()
        self.Expert = nn.Sequential(
            nn.Conv2d(out_channels,out_channels//2,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2,out_channels,kernel_size=3,padding=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),
            nn.Dropout(drop_path_rate)
        )
    def forward(self,x):
        return self.Expert(x)

class MoEDecoderModule(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, use_moe=True, expert_num=2, choice_moe='dmoe', drop_path_rate=0., learnable_vec=0.5 ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.moe = DifferenceMoE(expert=ExpertModule(out_channels,drop_path_rate),
                                expert_dim=out_channels, 
                                expert_num=expert_num,
                                ablation=choice_moe, 
                                learnable_vec=learnable_vec
                                ) if use_moe else ExpertModule(out_channels,drop_path_rate)
        

    def forward(self, x):
        x = self.proj(x)
        x = self.moe(x)
        return x

class FeatureDownScale(nn.Module):
    def __init__(self, down_scale, in_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, in_c, kernel_size=down_scale, stride=down_scale)
        self.norm = nn.BatchNorm2d(in_c)

    def forward(self, x):

        x = self.norm(self.proj(x))
        return x

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride, groups = dim_in,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)
    
class FFNModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 drop_path_rate=0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        hidden_dim = 4 * in_channels
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 1),
        )
    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        return self.drop_path(self.norm(self.ffn(x)+x))

class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', groups=out_channels)
        self.conv1x1end = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        x = self.conv1x1end(self.gelu(self.batchnorm(self.conv3x3(self.conv1x1(x)))))

        return x
    
class LightTransformerModule(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_rate=0., multi_k=6, use_moe=False, expert_num=2, choice_moe='dmoe', learnable_vec=-1):
        super().__init__()
        
        self.transformer = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding='same',),
                            nn.BatchNorm2d(out_channels),
                            MultiKVAttention(in_channels=out_channels, multi=multi_k),
                            DifferenceMoE(
                                expert=FFNModule(out_channels,drop_path_rate),
                                expert_dim=out_channels, 
                                expert_num=expert_num,
                                ablation=choice_moe, 
                                learnable_vec=learnable_vec
                            ) if use_moe else FFNModule(out_channels, drop_path_rate),
                            MultiKVAttention(in_channels=out_channels, multi=multi_k),
                            DifferenceMoE(
                                expert=FFNModule(out_channels,drop_path_rate),
                                expert_dim=out_channels, 
                                expert_num=expert_num,
                                ablation=choice_moe, 
                                learnable_vec=learnable_vec
                            ) if use_moe else FFNModule(out_channels, drop_path_rate),              
        )

    def forward(self, x):
        return self.transformer(x)


class DownAgriSegModule(nn.Module):
    """
    in_channels: input feature map channels
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, drop_path_rate=0., multi_k=6, expert_num=2, use_moe=False, choice_moe='dmoe', learnable_vec=-1):
        super().__init__()
        self.down = FeatureDownScale(down_scale=2, in_c=in_channels)
        self.module = LightTransformerModule(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             drop_path_rate=drop_path_rate, 
                                             multi_k=multi_k, 
                                             expert_num=expert_num,
                                             use_moe=use_moe, 
                                             choice_moe=choice_moe, 
                                             learnable_vec=learnable_vec)

    def forward(self, x):
        x = self.down(x)
        x = self.module(x)
        return x

class UpMoEModule(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, up_channel, skip_channels, out_channels, drop_path_rate=0.1, bilinear=True, use_moe=True, expert_num=2, choice_moe='dmoe',  learnable_vec=0.5):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_channel, up_channel, kernel_size=2, stride=2)

        self.module = MoEDecoderModule(
            in_channels=(skip_channels+up_channel), 
            out_channels=out_channels, 
            use_moe=use_moe, 
            expert_num=expert_num,
            choice_moe=choice_moe, 
            drop_path_rate=drop_path_rate, 
            learnable_vec=learnable_vec
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.module(x)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

class AgriSeg_Lighting(nn.Module):
    def __init__(self, n_classes, drop_path_rate=0.1, multi_k=6, expert_num=2, use_moe_encoder=False, use_moe_decoder=True, choice_moe='dmoe', learnable_vec=0.5):
        super().__init__()
        self.n_classes = n_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        self.emb = Embeddings(3, 64, patch_size=4, drop_rate=-1)
        self.inc = LightTransformerModule(64, 64,  drop_path_rate, multi_k=multi_k, use_moe=use_moe_encoder, expert_num=expert_num, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.down1 = DownAgriSegModule(64, 128, dpr[0], multi_k=multi_k, expert_num=expert_num, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.down2 = DownAgriSegModule(128, 256, dpr[1], multi_k=multi_k, expert_num=expert_num, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.down3 = DownAgriSegModule(256, 512, dpr[2], multi_k=multi_k, expert_num=expert_num, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.up1 = UpMoEModule(512, 256, 256, dpr[0], use_moe=use_moe_decoder, expert_num=expert_num, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.up2 = UpMoEModule(256, 128, 128, dpr[1], use_moe=use_moe_decoder, expert_num=expert_num, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.up3 = UpMoEModule(128, 64, 64, dpr[2], use_moe=use_moe_decoder, expert_num=expert_num, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.invemb = InverseEmbeddings(patch_size=4, in_channels=64, out_channels=64, drop_path_rate=dpr[3])
        self.outc = Out(64, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
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
        x = self.emb(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.invemb(x)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)
    net = AgriSeg_Lighting(n_classes=4, drop_path_rate=0.3, multi_k=6, expert_num=2, use_moe_encoder=False, use_moe_decoder=True, choice_moe='dmoe', learnable_vec=0.5).cuda()
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