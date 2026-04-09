# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
import torch.nn as nn
from Nets.backbone.bra_unet_system import BRAUnetSystem
logger = logging.getLogger(__name__)
class BRAUnet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1, n_win=8):
        super(BRAUnet, self).__init__()
        self.bra_unet = BRAUnetSystem(img_size=img_size,
                                      in_chans=in_chans,
                                      num_classes=num_classes,
                                      head_dim=32,
                                      n_win=n_win,
                                      embed_dim=[96, 192, 384, 768],
                                      depth=[2, 2, 8, 2],
                                      depths_decoder=[2, 8, 2, 2],
                                      mlp_ratios=[3, 3, 3, 3],
                                      drop_path_rate=0.2,
                                      topks=[2, 4, 8, -2],
                                      qk_dims=[96, 192, 384, 768])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.bra_unet(x)
        return logits
    def load_from(self):
        pretrained_path = '/home/caipengzhou/SA_Med_Seg/DCSAU-Net/pretrained_ckpt/biformer_base_best.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self.bra_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.bra_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")
if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)

    net = BRAUnet(img_size=512, in_chans=3, num_classes=2, ).cuda()
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

