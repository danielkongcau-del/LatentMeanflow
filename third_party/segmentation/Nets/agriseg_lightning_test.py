import torch
import torch.nn as nn
import torch.nn.functional as F
from Nets.cooperative_games_moe import CooperativeGamesMoE
from Nets.kv_attention import KVAttention, MultiKVAttention

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
    def __init__(self, in_channels=64, out_channels=64, patch_size=4, drop_rate=0., bilinear=True, use_moe=True, choice_moe='ab3', drop_path_rate=0., learnable_vec=0.5):
        super(InverseEmbeddings, self).__init__()
                # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=patch_size, stride=patch_size)
        #self.norm = nn.BatchNorm2d(out_channels) if not bilinear else nn.Identity()
        #self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
        self.module = MoEDecoderModule(
            in_channels=in_channels, out_channels=out_channels, 
            use_moe=use_moe, choice_moe=choice_moe, drop_path_rate=drop_path_rate, learnable_vec=learnable_vec)
        #self.module = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        #x = self.dropout(self.norm(self.up(x)))  
        x = self.up(x)
        x = self.module(x)
        return x

class SimpleInverseEmbeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_channels=64, out_channels=64, patch_size=4, drop_rate=0., use_moe=True, choice_moe='ab3', drop_path_rate=0., learnable_vec=0.5):
        super(SimpleInverseEmbeddings, self).__init__()
        self.patch_embeddings = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_path_rate)
        )


    def forward(self, x):
        x = self.dropout(self.norm(self.patch_embeddings(x)))  
        x = self.double_conv(x)
        return x

class ExpertModule(nn.Module):
    def __init__(self,out_channels,drop_path_rate=0.):
        super().__init__()
        # self.conv_in = nn.Conv2d(out_channels,out_channels//2,kernel_size=3,padding=1)
        # self.conv_out = nn.Conv2d(out_channels//2,out_channels,kernel_size=3,padding=1)
        # self.act = nn.ReLU(inplace=True),
        # self.norm_in = nn.BatchNorm2d(out_channels//2)
        # self.norm_out = nn.BatchNorm2d(out_channels)
        # self.drop = nn.Dropout(drop_path_rate)
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
        # x = self.act(self.norm_in(self.conv_in(x)))
        # x = self.act(self.norm_out(self.conv_out(x)))

        return self.Expert(x)

class MoEDecoderModule(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, use_moe=True, choice_moe='ab3', drop_path_rate=0., learnable_vec=0.5 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.moe = CooperativeGamesMoE(expert=ExpertModule(out_channels,drop_path_rate),
                                       expert_dim=out_channels, ablation=choice_moe, learnable_vec=learnable_vec) if use_moe else ExpertModule(out_channels,drop_path_rate)
        

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        #x = self.conv(self.bn(x))
        #x = self.conv(x)
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
            # nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
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
        # self.DCex = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, padding='same')
        # self.DConv = nn.Conv2d(4 * in_channels, 4 * in_channels, kernel_size=3, padding='same', groups=4 * in_channels)
        # self.DCre = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, padding='same')
        self.norm = nn.BatchNorm2d(in_channels)
        # self.gelu = nn.ReLU()
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
        # skip = x
        # x = self.DCre(self.drop_path(self.gelu(self.DConv(self.DCex(x)))))
        # return self.norm(x+skip)
        # x = self.DCre(self.gelu(self.DCex(self.norm(self.DConv(x)))))
        # return self.drop_path(x)+skip
        
class FFNModule_origin(nn.Module):
    def __init__(self,
                 in_channels: int,
                 drop_path_rate=0.
                 ):
        super().__init__()
        self.DConv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same', groups=in_channels)
        self.DCex = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, padding='same')
        self.DCre = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, padding='same')
        self.norm = nn.LayerNorm(in_channels)
        self.gelu = nn.GELU()
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        skip = x
        x = self.DCre(self.gelu(self.DCex(self.norm(self.DConv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))))
        return self.drop_path(x)+skip

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


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        b,c,h,w = x.shape
        x = x.reshape(b,c,-1).permute(0,2,1)
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, h, w, C).permute(0,3,1,2)
        return x

class SimpleAttention(nn.Module):
    """[1,1,c]->[1,c]@[c,c]->[1,c]*[h*w,c]->[h,w,c]"""
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 2
                 ):
        super().__init__()
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.scale: float = num_heads ** -0.5
        # Init layers
        self.norm = nn.LayerNorm(in_channels)
        self.v = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.a = nn.Linear(in_features=1, out_features=in_channels//num_heads, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.s = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding='same',groups=in_channels,bias=True)

    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        # [B,C,H,W]->[B,C,1,1]
        # [B,C,1,1]->[B,C,1]
        # x_temp: [B,C,1]
        skip = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_temp = x.mean((2, 3), keepdim=True).reshape(_B, _C, -1)
        # score_map: [B,C,C/head]
        # reshape:[B,num_head,c//num_head,C/head]
        Ka = self.a(x_temp).reshape(_B, self.num_heads, _C//self.num_heads, -1)

        # x_temp:[B,C,1] -> permute X_K [B,1,C]
        # x_k: [B,1,C]
        Kv = x_temp.permute(0, 2, 1)
        # reshape:[B,1,num_head,embed_dim_per_head]
        # permute:[B,num_head,1,embed_dim_per_head]
        Kv = self.v(Kv).reshape(_B, 1, self.num_heads, -1).permute(0, 2, 1, 3)

        # score:[B,num_head,1,C/head]
        # transpose:[B,num_head,C/head,1]
        # reshape:[B,C,1,1]
        score = self.softmax(Kv @ Ka.transpose(-2, -1) * self.scale).transpose(1, 2).reshape(_B, -1, 1, 1)
        # x[B,C,H,W]
        # score[B,C,1,1]
        x = self.s(x) * score
        # x[B,C,H,W]
        return x + skip
    
class LightTransformerModule(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_rate=0., choice_attn='no_norm_no_act', use_moe=False, choice_moe='ab3', learnable_vec=-1):
        super().__init__()
        
        self.transformer = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding='same',),
                            nn.BatchNorm2d(out_channels),
                            MultiKVAttention(in_channels=out_channels, ablation=choice_attn, multi=6),
                            CooperativeGamesMoE(
                            FFNModule(out_channels,drop_path_rate),
            expert_dim=out_channels, ablation=choice_moe, learnable_vec=learnable_vec
        ) if use_moe else FFNModule(out_channels, drop_path_rate),
                            MultiKVAttention(in_channels=out_channels, ablation=choice_attn, multi=6),
                            CooperativeGamesMoE(
                            FFNModule(out_channels,drop_path_rate),
            expert_dim=out_channels, ablation=choice_moe, learnable_vec=learnable_vec
        ) if use_moe else FFNModule(out_channels, drop_path_rate),
                            
        )

    def forward(self, x):
        # CNN block
        #cnn_shortcut = x
        #x = self.cnnblock(x)
        #x = self.drop_path(x) + self.skip_path(cnn_shortcut)
        # Simple-former block
        #x = self.ffn(x)
        #x = self.attention(x)
        
        return self.transformer(x)

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.norm(self.to_out(out)+x)

class TransformerModule(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_rate=0., choice_attn='no_norm_no_act', use_moe=False, choice_moe='ab3', learnable_vec=-1):
        super().__init__()
        self.transformer = nn.ModuleList([
            nn.Sequential(
                EfficientSelfAttention(dim=in_channels, heads=4, reduction_ratio=1),
                CooperativeGamesMoE(
                    FFNModule(out_channels, drop_path_rate),
                    expert_dim=out_channels,
                    ablation=choice_moe,
                    learnable_vec=learnable_vec
                ) if use_moe else FFNModule(out_channels, drop_path_rate)
            )
            for i in range(2)
        ])
    
        

    def forward(self, x):
        for module in self.transformer:
            x = module(x)
        return x



class DownAgriSegModule(nn.Module):
    """
    in_channels: input feature map channels
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, drop_path_rate=0., choice_attn='no_norm_no_act', use_moe=False, choice_moe='ab3', learnable_vec=-1):
        super().__init__()
        self.down = FeatureDownScale(down_scale=2, in_c=in_channels)
        self.module = LightTransformerModule(in_channels=in_channels, out_channels=out_channels, drop_path_rate=drop_path_rate, choice_attn=choice_attn, 
                                             use_moe=use_moe, choice_moe=choice_moe, learnable_vec=learnable_vec)

    def forward(self, x):
        x = self.down(x)
        x = self.module(x)
        return x

class UpMoEModule(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, up_channel, skip_channels, out_channels, drop_path_rate=0.1, bilinear=True, use_moe=True, choice_moe='ab3',  learnable_vec=0.5):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_channel, up_channel, kernel_size=2, stride=2)

        self.module = MoEDecoderModule(in_channels=(skip_channels+up_channel), out_channels=out_channels, use_moe=use_moe, choice_moe=choice_moe, drop_path_rate=drop_path_rate, learnable_vec=learnable_vec)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
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
    def __init__(self, n_classes, drop_path_rate=0.1, choice_attn='no_norm_no_act', use_moe_encoder=False, use_moe_decoder=True, choice_moe='ab3', learnable_vec=0.5):
        super().__init__()
        self.n_classes = n_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        self.emb = Embeddings(3, 64, patch_size=4, drop_rate=-1)
        self.inc = LightTransformerModule(64, 64,  drop_path_rate, choice_attn=choice_attn, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.down1 = DownAgriSegModule(64, 128, dpr[0], choice_attn=choice_attn, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.down2 = DownAgriSegModule(128, 256, dpr[1], choice_attn=choice_attn, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.down3 = DownAgriSegModule(256, 512, dpr[2], choice_attn=choice_attn, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        # self.bottleneck = TransformerModule(512, 512, dpr[3], choice_attn=choice_attn, use_moe=use_moe_encoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.up1 = UpMoEModule(512, 256, 256, dpr[0], use_moe=use_moe_decoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.up2 = UpMoEModule(256, 128, 128, dpr[1], use_moe=use_moe_decoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.up3 = UpMoEModule(128, 64, 64, dpr[2], use_moe=use_moe_decoder, choice_moe=choice_moe, learnable_vec=learnable_vec)
        self.invemb = InverseEmbeddings(patch_size=4, in_channels=64, out_channels=64, drop_rate=-1, use_moe=False, choice_moe=choice_moe, drop_path_rate=dpr[3])
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
        # x4 = self.bottleneck(x4)
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
    net = AgriSeg_Lighting(n_classes=4, drop_path_rate=0.3, choice_attn='no_norm_no_act', use_moe_encoder=True, use_moe_decoder=True, choice_moe='ab3', learnable_vec=-1).cuda()
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

    # throughput
    batch_size = 2
    num_batches = 80
    input_data = torch.randn(1, 3, 512, 512).cuda()
    def process_image(image):
        return net(image)
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in range(num_batches):
            output = process_image(input_data)
    end_time = time.time()
    total_images_processed = batch_size * num_batches
    throughput = total_images_processed / (end_time - start_time)
    print(f"throughput: {throughput:.2f} Image/s")