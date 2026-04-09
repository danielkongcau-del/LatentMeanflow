import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np
from einops import rearrange
from math import floor

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def patchify_unfold(imgs, kernel_size, stride=1, padding=0, dilation=1):
    """
    imgs: (N, C, H, W)
    p: patch_size
    x: (N x h x w, C, P, P)
    """
    N, C, H, W = imgs.shape
    h = floor(((H + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    # w = floor(((W + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    x = torch.nn.functional.unfold(imgs, kernel_size=(kernel_size, kernel_size), stride=stride, dilation=dilation, padding=padding)
    x = rearrange(x, 'b (c ph pw) (h w) -> (b h w) c ph pw', ph=kernel_size, pw=kernel_size, h=h)
    return x
def patchify_enlarged(imgs, patch_size, context_padding, padding_mode='replicate'):
    kernel_size = patch_size + context_padding * 2
    # patche2 = patchify_unfold(imgs, kernel_size=kernel_size, stride=patch_size, padding=context_padding)
    imgs_pad = torch.nn.functional.pad(imgs, (context_padding, context_padding, context_padding, context_padding), mode=padding_mode)
    patches = patchify_unfold(imgs_pad, kernel_size=kernel_size, stride=patch_size)
    return patches


def remove_padding(imgs, padding):
    return imgs[..., padding:-padding, padding:-padding]

def unpatchify(x, batch_size, context_padding=0):
    """
    x: ((N h w), C, patch_size, patch_size)
    imgs: (N, C, H, W)
    """
    h = w = int((x.shape[0]/batch_size)**.5)
    assert h * w == x.shape[0]/batch_size
    
    if context_padding > 0:
        x = remove_padding(x, padding=context_padding)
    imgs = rearrange(x, '(b h w) c ph pw -> b c (h ph) (w pw)', h=h, w=w)
    return imgs

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class PatchBlock(nn.Module):
    def __init__(self, img_size, patch_size, depths, in_chans, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop, drop_path_rates, norm_layer, sr_ratio):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
                
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths))]  # stochastic depth decay rule
        self.block = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_path_rates[i], norm_layer=norm_layer,
            sr_ratio=sr_ratio)
            for i in range(depths)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for i, blk in enumerate(self.block):
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # [B,C,H,W]
        return x


class PatchFormerBlock(nn.Module):
    def __init__(self, img_size, large_patch, context_padding, patch_size, depths, in_chans, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop, drop_path_rates, norm_layer, sr_ratio, pos_embed, alt):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=large_patch + 2 * context_padding, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
                
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths))]  # stochastic depth decay rule
        self.block = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_path_rates[i], norm_layer=norm_layer,
            sr_ratio=sr_ratio)
            for i in range(depths)])
        self.norm = norm_layer(embed_dim)
        self.large_patch = large_patch
        self.context_padding = context_padding
        self.patch_size = patch_size
        self.alt = alt

        if pos_embed:
            assert embed_dim is not None
            num_patch = int(img_size/large_patch)**2
            self.register_buffer('pos_embed', torch.zeros(num_patch, 1, embed_dim))
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patch**.5), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(1))
        else:
            self.pos_embed = None


    def forward(self, x):
        img = x
        x = patchify_enlarged(x, self.large_patch, context_padding=self.context_padding)
        B = x.shape[0]

        x, H, W = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed.repeat((img.shape[0], 1, 1))
        for i, blk in enumerate(self.block):
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = unpatchify(x, img.shape[0], context_padding=int(self.context_padding/self.patch_size))

        return x


class PatchTransformer(nn.Module):
    def __init__(self, img_size=224, patch_block_type='patchformer', large_patch=[64,32,16,32], context_padding=[2,2,2,2], patch_sizes=[4, 4, 4, 4],
                 in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pos_embed=False, alt=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_block_type = patch_block_type

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cum_depth = np.cumsum(depths)

        self.encoder = nn.ModuleList()
        for i in range(len(depths)):
            drop_path_rate_arr = dpr[0 if i == 0 else cum_depth[i - 1]: cum_depth[i]]
            in_dims = in_chans if i == 0 else embed_dims[i - 1]
            if patch_block_type == 'patchformer':
                block_alt = False if alt is None else alt[i]
                encoder_module = PatchFormerBlock(img_size, large_patch[i], context_padding[i], patch_sizes[i], depths[i], 
                                            in_dims, embed_dims[i], num_heads[i], mlp_ratios[i], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, drop_path_rate_arr, norm_layer, sr_ratios[i], pos_embed, block_alt)
            elif patch_block_type == 'patchblock':
                encoder_module = PatchBlock(img_size, patch_sizes[i], depths[i], 
                                            in_dims, embed_dims[i], num_heads[i], mlp_ratios[i], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, drop_path_rate_arr, norm_layer, sr_ratios[i])
            else:
                raise ValueError('unknow patch_block_type!')
            
            img_size /= patch_sizes[i]
                
            self.encoder.append(encoder_module)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # x: [B, 2, 256, 256]

        # stage 1
        # x = patchify_enlarged(x, self.large_patch, context_padding=self.context_padding)
        # x = self.patch_embed(x)
        # H, W = x.shape[2], x.shape[3]
        # for i, blk in enumerate(self.block1):
        #     x = blk(x, H, W)
        # x = self.norm1(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        # x: [B, 32, 192, 192]
        
        for encoder_module in self.encoder:
            x = encoder_module(x)
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x



# MoE expert 
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), final_act=True, activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.final_act = final_act
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for affine in self.affine_layers:
            x = affine(x)
            if affine != self.affine_layers[-1] or self.final_act:
                x = self.activation(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x

class MOEHead(nn.Module):
    """
        Moe head
    """
    def __init__(self, num_classes, in_channels, embed_dim, feature_strides, prescale_mlp_dims=None, prescale_mlp_final_act=True,
                 afterscale_mlp_dims=[512, 256], afterscale_mlp_final_act=True, moe_mlp_dims=[512, 256], moe_conv_dims=None, activation='relu', use_linear_fuse=True, dropout_ratio=0.1, **kwargs):
        super(MOEHead, self).__init__()
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.prescale_mlp_dims = prescale_mlp_dims
        self.afterscale_mlp_dims = afterscale_mlp_dims
        self.use_linear_fuse = use_linear_fuse

        embedding_dim = embed_dim
        self.in_channels = in_channels
        self.num_classes = num_classes

        cur_dim = sum(self.in_channels)
        if prescale_mlp_dims is not None:
            self.prescale_mlp = nn.ModuleList()
            for in_channel in self.in_channels:
                mlp = MLP(in_channel, prescale_mlp_dims, prescale_mlp_final_act, activation)
                self.prescale_mlp.append(mlp)

        cur_dim = len(self.in_channels) * prescale_mlp_dims[-1]

        if moe_conv_dims is not None:
            self.moe_conv = nn.ModuleList()
            conv_dims = moe_conv_dims + [len(self.in_channels)]
            for conv_dim in conv_dims:
                conv_layer = nn.Sequential(
                    nn.Conv2d(
                    in_channels=cur_dim,
                    out_channels=conv_dim,
                    kernel_size=3, stride=1, padding=1,
                    ),
                    nn.BatchNorm2d(conv_dim),
                )
                cur_dim = conv_dim
                self.moe_conv.append(conv_layer)
        else:
            self.moe_conv = None


        if moe_mlp_dims is not None:
            self.moe_mlp = MLP(cur_dim, moe_mlp_dims + [len(self.in_channels)], False, activation)
        else:
            self.moe_mlp = None

        if afterscale_mlp_dims is not None:
            self.afterscale_mlp = MLP(prescale_mlp_dims[-1], afterscale_mlp_dims, afterscale_mlp_final_act, activation)
        cur_dim = afterscale_mlp_dims[-1]
        
        if use_linear_fuse:
            self.linear_fuse = nn.Sequential(
                    nn.Conv2d(
                        in_channels=cur_dim,
                        out_channels=embedding_dim,
                        kernel_size=1, 
                    ),
                    nn.BatchNorm2d(embedding_dim),
                )
            cur_dim = embedding_dim

        self.linear_pred = nn.Conv2d(cur_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout_ratio)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs):
        # x = self._transform_inputs(inputs)
        x = inputs
        largest_size = x[0].shape[-2:]
        x_scaled = []
        for i, x_i in enumerate(x):
            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)

            if x_i.shape[-2:] != largest_size:
                x_i_scaled = torch.nn.functional.interpolate(x_i, size=largest_size, mode='bilinear', align_corners=False)
            else:
                x_i_scaled = x_i

            x_scaled.append(x_i_scaled)

        x_stacked = torch.stack(x_scaled, dim=1) 
        x = torch.cat(x_scaled, dim=1)

        if self.moe_conv is not None:
            for conv_layer in self.moe_conv:
                x = conv_layer(x)

        if self.moe_mlp is not None:
            x = self.moe_mlp(x)

        moe_weights = torch.softmax(x, dim=1)
        x = (x_stacked * moe_weights.unsqueeze(2)).sum(1)

        if self.afterscale_mlp_dims is not None:
            x = self.afterscale_mlp(x)

        if self.use_linear_fuse:
            x = self.linear_fuse(x)

        x = self.dropout(x)
        x = self.linear_pred(x)
        x = self.upsampling(x)
        # if img_metas is not None:
        #     case = img_metas[0]['filename'].split('/')[-1].split('.')[0]
        #     save_dir = 'results/moe_weights_cmap'
        #     weights = moe_weights.cpu().numpy()
        #     for i in range(moe_weights.shape[1]):
        #         w = weights[0,i,:,:]
        #         filename = f'{save_dir}/{case}_{i}.png'
        #         plt.imsave(filename, w, cmap='OrRd', vmin=0, vmax=0.6)
        #         # cv2.imwrite(filename, w*255)
        return x
        
class Patcher(nn.Module):
    def __init__(self, img_size, in_chans=3, n_classes=4 ):
        super().__init__()
        self.encoder = PatchTransformer(
            img_size=img_size, 
            patch_block_type='patchformer', 
            large_patch=[32,32,32,32], 
            context_padding=[8,8,8,8],
            patch_sizes=[2,2,2,2], 
            in_chans=in_chans, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, 
            depths=[3, 6, 40, 3], 
            sr_ratios=[2, 2, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1
        )
        self.decoder = MOEHead(
            in_channels=[64, 128, 320, 512],
            embed_dim=256,
            feature_strides=[4, 8, 16, 32],
            prescale_mlp_dims=[256, 256], # new
            prescale_mlp_final_act=True,
            afterscale_mlp_dims=[256, 256], # new
            afterscale_mlp_final_act=True,
            moe_mlp_dims=None, # new
            moe_conv_dims=[256],
            num_classes=n_classes,
            dropout_ratio=0.1,
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def config_patcher(img_size=512, n_classes=4):
    return Patcher(img_size=img_size, in_chans=3, n_classes=n_classes).cuda()

if __name__ == "__main__":
    from thop import profile
    import time

    print(torch.__version__)
    net = Patcher(img_size=512, in_chans=3, n_classes=4).cuda()
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