import torch
import torch.nn as nn
import torch.nn.functional as F


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

class FFNModule(nn.Module):
    def  __init__(self,
                 in_channels: int,
                 drop_path_rate=0.
                 ):
        super().__init__()
        self.DConv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same', groups=in_channels)
        self.DCex = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, padding='same')
        self.DCre = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, padding='same')
        self.norm = nn.LayerNorm(in_channels)
        self.gelu = nn.GELU()
        # defeat over-fitting
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()


    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        skip = x
        x = self.DCre(self.gelu(self.DCex(self.norm(self.DConv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))))
        return self.drop_path(x)+skip

class MoE_FFN(nn.Module):
    def __init__(self,in_channels:int, drop_path_rate=0., moe_choice='ab1' ):
        super(MoE_FFN,self).__init__()
        self.E_1 = FFNModule(in_channels, drop_path_rate)
        self.E_2 = FFNModule(in_channels, drop_path_rate)
        self.E_3 = FFNModule(in_channels, drop_path_rate)
        self.alpha = nn.Parameter(torch.randn(1,requires_grad=True))
        self.choice = moe_choice
    def forward(self,x):
        if self.choice == 'ab1':
            e1_out = self.E_1(x)
            e2_out = self.E_2(x)
            weighted_sum = e1_out - self.alpha * e2_out
            x = F.softmax(weighted_sum, dim=1)
            x = self.E_3(x)
        elif self.choice == 'ab2':
            e1_out = self.E_1(x)
            e2_out = self.E_2(x)
            x_1 = F.softmax(e1_out, dim=1)
            s_2 = self.alpha * e2_out
            x_2 = F.softmax(s_2, dim=1)
            x = x_1 - x_2
            x = self.E_3(x)
        else:
            e1_out = self.E_1(x)
            e2_out = self.E_2(x)
            s_2 = self.alpha * e2_out
            x = e1_out - s_2
            x = self.E_3(x)
        return x

if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)
    net = MoE_FFN(in_channels=64, moe_choice='ab3').cuda()
    print(net)
    image = torch.rand(1, 64,256, 256).cuda()

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
    # 模拟输入数据，例如一批图像
    batch_size = 2  # 每批处理的图像数量
    num_batches = 80  # 总共处理的批次数量
    input_data = torch.randn(batch_size, 64,256, 256).cuda()  # 模拟输入图像数据


    # 模拟一个简单的前向传播
    def process_image(image):
        return net(image)


    # 将模型切换到评估模式
    net.eval()
    # 计时开始
    start_time = time.time()
    # 模拟处理多批图像
    with torch.no_grad():
        for batch in range(num_batches):
            output = process_image(input_data)  # 调用模型处理图像
    # 计时结束
    end_time = time.time()
    # 计算吞吐量（每秒处理的图像数量）
    total_images_processed = batch_size * num_batches
    throughput = total_images_processed / (end_time - start_time)
    print(f"模型吞吐量: {throughput:.2f} 图像/秒")