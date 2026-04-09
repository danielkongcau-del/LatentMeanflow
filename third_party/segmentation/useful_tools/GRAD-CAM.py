import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from Nets.UNet import UNet
import torch
import os
from glob import glob
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from Nets.patchformer import config_patcher
from pytorch_grad_cam import GradCAM
from Nets.swinsmt import SwinSMT,config_swinsmt
from Nets.segformer import segformermodel
from Nets.transunet import config_transunet
from Nets.patchformer import config_patcher
import cv2
from Nets.SegNet import SegNet
from Nets.munet import MUNet

from Nets.UNet import UNet
from Nets.swinmoba import SwinMoBA_Seg
from Nets.nuseg_moe import config_transnuseg,TransNuSeg
from Nets.agriseg_lighting import AgriSeg_Lighting
from Nets.transunet import config_transunet
from Nets.nuseg_moe import config_transnuseg
from Nets.deeplabv3 import DeepLab
from Nets.cpunet import CPUNet
from Nets.AttentionUnet import SampleOrMHSAUNet

# 图像路径
from Nets.CSNet import CSNet
image_path = r"E:\13_张硕华\pest权重\DMOE热力图\output4.jpg"
image = np.array(Image.open(image_path).convert('RGB'))
# image = np.array(cv2.imread(image_path))
# print(image)

rgb_img = np.float32(image) / 255

# 预处理图像
preprocess = transforms.Compose([
    transforms.ToTensor(),
        ])
input_tensor = preprocess(image).unsqueeze(0)

# 模型定义和权重加载
model = config_patcher(img_size=512, n_classes=2)
pretrained_weights_path = r'E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\work_dir\seg\pest\aug-patcher-h512w512-erode0-weighting_none\patcher_best.pt'
msg = model.load_state_dict(torch.load(pretrained_weights_path, map_location='cpu'))
print(msg)
model.eval()

# 将模型和输入张量移动到 GPU（如果可用）
if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

# 定义包装类以适配模型输出
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

model = SegmentationModelOutputWrapper(model)

# 获取模型输出并归一化
with torch.no_grad():
    output = model(input_tensor)
normalized_masks = F.softmax(output, dim=1).cpu()

# 类别定义
sem_classes = ['__background__', '1','2','3']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

# 假设我们对 'aeroplane' 类别感兴趣
target_category = sem_class_to_idx['1']

# 创建目标掩码
car_mask = normalized_masks[0].argmax(dim=0).detach().cpu().numpy()
car_mask_float = np.float32(car_mask == target_category)

# 定义目标类别的热力图计算类
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

# 设置目标层
target_layers = [model.model.decoder]  # 根据你的UNet结构调整 model.model[-1]

# 创建GradCAM实例并生成热力图
targets = [SemanticSegmentationTarget(target_category, car_mask_float)]

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 显示或保存结果
output_image = Image.fromarray((cam_image * 255).astype(np.uint8))
output_image.show()  # 或者使用 output_image.save('path_to_save.png') 保存图像
output_image.save(r'E:\13_张硕华\pest权重\DMOE热力图\pest\patcher\4.jpg')
