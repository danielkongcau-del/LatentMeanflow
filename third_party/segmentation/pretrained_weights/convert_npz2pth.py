import numpy as np
import torch

def npz_to_pth(npz_filepath, pth_filepath):
    # 加载 .npz 文件
    with np.load(npz_filepath) as data:
        # 创建一个字典来存储张量
        tensor_dict = {}
        for key in data.files:
            # 将 numpy 数组转换为 PyTorch 张量并添加到字典中
            tensor_dict[key] = torch.from_numpy(data[key])
    
    # 保存为 .pth 文件
    torch.save(tensor_dict, pth_filepath)
    print(f"Data has been successfully converted and saved to {pth_filepath}")

# 使用函数进行文件转换
npz_filepath = './vit_base_patch16_384.npz'  # 输入的 .npz 文件路径
pth_filepath = './vit_base_patch16_384.pth'  # 输出的 .pth 文件路径
npz_to_pth(npz_filepath, pth_filepath)