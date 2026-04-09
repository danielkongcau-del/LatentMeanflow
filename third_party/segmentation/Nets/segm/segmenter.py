import torch
import torch.nn as nn
import torch.nn.functional as F

from Nets.segm.utils import padding, unpadding
from Nets.segm.vit import VisionTransformer
from Nets.segm.decoder import MaskTransformer,DecoderLinear
import torch.utils.checkpoint as checkpoint

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        use_checkpoint=True,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.use_checkpoint = use_checkpoint

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params
    def forward(self,im):
        if self.use_checkpoint:
            masks = checkpoint.checkpoint(self._forward, im)
        else:
            masks = self._forward(im)
        return masks
    def _forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

def segmentermodel(num_classes,use_checkpoint=True):
    return Segmenter(encoder=VisionTransformer(image_size=(512, 512),
                            patch_size=4,
                            n_layers=12,
                            d_model=384,
                            d_ff=384*4,
                            n_heads=1,
                            n_cls=num_classes,
                            dropout=0.0,
                            drop_path_rate=0.0,
                            distilled=False,
                            channels=3
        ),
        decoder=MaskTransformer(
                n_cls=num_classes,
                patch_size=4,
                d_encoder=384,
                n_layers=2,
                n_heads=1,
                d_model=384,
                d_ff=384*4,
                drop_path_rate=0.1,
                dropout=0.,
                # expert_num=2,
                # learnable_vec=-1,
                # use_moe=True,
                # ablation='dmoe'
        ),
        n_cls=num_classes,use_checkpoint=use_checkpoint
                     )

if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)
    net = segmentermodel(2).cuda()
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

    # 模拟输入数据，例如一批图像
    batch_size = 2  # 每批处理的图像数量
    num_batches = 80  # 总共处理的批次数量
    input_data = torch.randn(batch_size, 3, 800, 600).cuda()  # 模拟输入图像数据


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