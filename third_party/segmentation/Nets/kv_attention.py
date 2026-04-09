import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision
    
class MultiK(nn.Module):
    def __init__(self, in_channels, multi=4):
        super().__init__()
        self.num_scale = multi

        if self.num_scale > 1:
            for i in range(self.num_scale):
                multiscale_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                            padding=(1+i), stride=1, groups=in_channels, dilation=(1+i))
                setattr(self, f"multiscale_conv_{i + 1}", multiscale_conv)
        else:
            multiscale_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                            padding=1, stride=1, groups=in_channels)
            setattr(self, f"multiscale_conv_1", multiscale_conv)
        self.proj = nn.Conv2d(multi * in_channels, in_channels, kernel_size=1) 
        self.norm = nn.BatchNorm2d(multi * in_channels)
        self.act = nn.GELU()
    def forward(self, x):
        
        if self.num_scale > 1:
            for i in range(self.num_scale):
                multiscale_conv = getattr(self, f"multiscale_conv_{i + 1}")
                multi_k = multiscale_conv(x)
                if i == 0:
                    multi_k_out = multi_k
                else:
                    multi_k_out = torch.cat([multi_k_out, multi_k], 1)
        else:
            multiscale_conv = getattr(self, f"multiscale_conv_1")
            multi_k_out = multiscale_conv(x)

        out = self.proj(self.norm(multi_k_out))

        return out 
    
class MultiKVAttention(nn.Module):
    def __init__(self, in_channels, multi=6):
        super().__init__()
        self.k = MultiK(in_channels=in_channels, multi=multi)
        self.v = nn.Conv2d(in_channels=in_channels, 
                           out_channels=in_channels, 
                           kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        # [B, C, H, W]
        k = self.k(x)
        v = self.v(x)
        out = (k * v) + x
            
        return self.norm(out)     


# ----------------------------------------test--------------------------------------------------------------

class KVAttention(nn.Module):
    def __init__(self, in_channels, ablation='no_norm_no_act'):
        super().__init__()
        self.k = nn.Conv2d(in_channels=in_channels, 
                           out_channels=in_channels, 
                           kernel_size=3,
                           padding=1,
                        #    groups=in_channels
                                   )
        self.v = nn.Conv2d(in_channels=in_channels, 
                           out_channels=in_channels, 
                           kernel_size=1)
        self.proj = nn.Conv2d(in_channels=in_channels, 
                           out_channels=in_channels, 
                           kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels) if ablation.find('no_norm') != -1 else nn.Identity()
        self.act = nn.GELU()
        self.ablation = ablation
    def forward(self, x):
        # [B, C, H, W]
        k = self.k(x)
        v = self.v(x)
        if self.ablation == 'norm_act_v':
            out = self.proj(self.norm(k * self.act(v)))+x
            print('norm_act_v')
        elif self.ablation == 'norm_act_k':
            out = self.proj(self.norm(self.act(k) * v))+x
            print('norm_act_k')
        elif self.ablation == 'norm_no_act':
            out = self.proj(self.norm(k * v))+x
            print('norm_no_act')
        elif self.ablation == 'no_norm_act_v':            
            out = self.proj(k * self.act(v))+x
            print('no_norm_act_v')
        elif self.ablation == 'no_norm_act_k':            
            out = self.proj(self.act(k) * v)+x
            print('no_norm_act_k')
        elif self.ablation == 'no_norm_no_act':
            out = self.proj(k * v)+x
            print('no_norm_no_act')
        return out
    



class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # offset的shape为(1,1,2*N,w,h) # 其中N=k_size*k_size; w,h是input_fea的宽高
        # 对offset的理解：offset保存的是，卷积核在input_fea上滑动时，以每个像素点为中心点时，所要聚合的邻居点的坐标索引修正值，这也是为什么每个像素点对应有2*N个通道（前N为x坐标，后N为y坐标）
        offset = self.p_conv(x)
        # print("offset:", offset.size())
        # 在卷积的乘加操作里，引入额外的权重（默认不用）
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # 获取滑动卷积的过程中，每次卷积的卷积核中心点的邻接点的索引(叠加了offset之后)
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # ============= 双线性插值计算浮点数坐标处的像素值 ===============
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # 这里的x_offset的含义是，卷积核在inp_fea上滑动卷积时，以inp每个点为中心点时，卷积核各点对应的像素的像素值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        # print(x_offset.size())
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 卷积核中每个点相对于卷积核中心点的偏移
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # 卷积核在特征图上滑动的中心点
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


if __name__ == "__main__":
    from thop import profile
    import time

    print(torch.__version__)
    net = KVAttention(in_channels=64, num_heads=1).cuda()
    print(net)
    image = torch.rand(1, 64, 64, 64).cuda()
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
    input_data = torch.randn(1, 64, 64, 64).cuda()
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