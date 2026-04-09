import warnings

warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from PIL import Image

import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms

# Grad-CAM 相关库
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

# 自定义模型导入

from choices import choose_net

# 加载模型和图像
def load_model_and_image(img_path, model_name, out_channels,pretrained_weights_path,way="cv2"):
    # 图像加载方式选择
    if way == "cv2":
        image = np.array(cv2.imread(img_path))  # 使用 OpenCV 读取 RGB 图像
    else:
        image = np.array(cv2.imread(image_path))

    # 调整图像大小至 512x512
    resize_transform = transforms.Compose([
        transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
        transforms.Resize((512, 512)),  # 调整大小到512x512
        transforms.ToTensor(),  # 转换为tensor
    ])

    input_tensor = resize_transform(image).unsqueeze(0)  # 添加batch维度
    rgb_img = np.float32(cv2.resize(image, (512, 512))) / 255  # 归一化到 [0, 1] 并调整大小

    # 模型构建和权重加载
    model = choose_net(model_name, out_channels)
    pretrained_weights_path = pretrained_weights_path
    msg = model.load_state_dict(torch.load(pretrained_weights_path, map_location='cpu'), strict=False)
    print(f"Model loaded: {msg}")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    return model, input_tensor, rgb_img


# 包装模型以适配 Grad-CAM 接口
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# 主函数：Grad-CAM 可视化
# 之前的代码保持不变，直到进入 visualize_gradcam 函数

# 修改 visualize_gradcam 函数中 GradCAM 的使用方式

def visualize_gradcam(model, input_tensor, rgb_img, target_category=1):
    wrapped_model = SegmentationModelOutputWrapper(model)

    # 获取模型输出
    with torch.no_grad():
        output = wrapped_model(input_tensor)
    normalized_masks = F.softmax(output, dim=1).cpu()

    # 获取语义分割掩码
    sem_classes = ['__background__', '1', '2', '3']  # 类别名
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    target_category_idx = sem_class_to_idx[str(target_category)]

    car_mask = normalized_masks[0].argmax(dim=0).detach().cpu().numpy()
    car_mask_float = np.float32(car_mask == target_category_idx)

    # 设置目标层
    target_layers = [wrapped_model.model.outc]  # 根据你的网络结构调整

    # 定义目标
    targets = [SemanticSegmentationTarget(target_category_idx, car_mask_float)]

    # 使用 GradCAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image

# 确保 SemanticSegmentationTarget 类正确实现了 __call__ 方法，返回一个标量
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # 计算指定类别的输出与掩码的乘积之和，产生一个标量
        return (model_output[self.category, :, :] * self.mask).sum()

# 其余部分保持不变...


# 执行主流程
if __name__ == '__main__':
    # 图像路径 & 模型参数配置
    image_path = r'E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\Image\disease1.jpg'  # 替换为你的图像路径
    model_name = 'segformer' #unet deeplabv3-resnet  segformer transunet  munet csnet  swin_unet
    out_channels = 2  # 根据你的任务调整类别数
    output_dir = r'E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\Image\disease'  # 热力图输出目录
    os.makedirs(output_dir, exist_ok=True)
    pretrained_weights_path = r"E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\work_dir\mpcr\disease\mpcr_extend_experiment\aug-csnet-h512w512-erode0-weighting_none\csnet_best.pt"
    image_open_way = 'cv2'
    model, input_tensor, rgb_img = load_model_and_image(image_path, model_name, out_channels,
                                         pretrained_weights_path,way=image_open_way)

    # 可视化指定类别（如 1）
    cam_image = visualize_gradcam(model, input_tensor, rgb_img, target_category=1)

    # 显示和保存结果
    result_image = Image.fromarray(cam_image)
    result_image.show()  # 展示图像
    result_image.save(os.path.join(output_dir, os.path.basename(image_path)))