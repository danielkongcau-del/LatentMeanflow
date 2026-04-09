import os
import shutil
import random
from pathlib import Path

def create_segmentation_dataset(
    image_dir,
    mask_dir,
    dataset_name,
    output_root_path,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    image_extensions=None
):
    """
    构建语义分割数据集，按比例划分 train/val/test，并保持 images 和 masks 文件名一一对应。
    
    Args:
        image_dir (str): 原始图像文件夹路径
        mask_dir (str): 原始掩码文件夹路径
        dataset_name (str): 输出数据集的文件夹名称
        output_root_path (str): 输出根目录路径（新数据集将保存在该目录下）
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例（自动补足剩余）
        seed (int): 随机种子，保证可复现
        image_extensions (list): 支持的图像格式
    """
    
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

    # 转为小写便于比较
    image_extensions = [ext.lower() for ext in image_extensions]
    
    # 设置随机种子
    random.seed(seed)
    
    # 转为 Path 对象
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    output_root = Path(output_root_path) / dataset_name  # 完整输出路径
    
    # 检查输入路径是否存在
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_path}")
    
    # 获取所有图像和掩码文件
    image_files = {
        f.stem: f for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    }
    mask_files = {
        f.stem: f for f in mask_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    }
    
    # 获取共有文件名（确保 image 和 mask 一一对应）
    common_stems = set(image_files.keys()) & set(mask_files.keys())
    if len(common_stems) == 0:
        raise ValueError("No matching image-mask pairs found.")
    
    print(f"Found {len(common_stems)} image-mask pairs.")
    
    # 转为列表并打乱
    file_list = sorted(common_stems)  # 排序以保证可复现
    random.shuffle(file_list)
    
    # 计算划分数量
    total = len(file_list)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val  # 剩余给 test
    
    print(f"Splitting into: train={n_train}, val={n_val}, test={n_test}")
    
    # 创建输出目录
    (output_root / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'train' / 'masks').mkdir(parents=True, exist_ok=True)
    (output_root / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'val' / 'masks').mkdir(parents=True, exist_ok=True)
    (output_root / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'test' / 'masks').mkdir(parents=True, exist_ok=True)
    
    def copy_files(file_stems, phase):
        for stem in file_stems:
            # 复制 image
            img_src = image_files[stem]
            img_dst = output_root / phase / 'images' / img_src.name
            shutil.copy2(img_src, img_dst)
            
            # 复制 mask
            mask_src = mask_files[stem]
            mask_dst = output_root / phase / 'masks' / mask_src.name
            shutil.copy2(mask_src, mask_dst)
        print(f"Copied {len(file_stems)} files to {phase}/")
    
    # 分配数据
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    
    # 执行复制
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"\n✅ Dataset created at: {output_root.resolve()}")
    print("Directory structure:")
    print(f"""
{output_root}/
├── train/
│   ├── images/  ({len(train_files)} files)
│   └── masks/   ({len(train_files)} files)
├── val/
│   ├── images/  ({len(val_files)} files)
│   └── masks/   ({len(val_files)} files)
└── test/
    ├── images/  ({len(test_files)} files)
    └── masks/   ({len(test_files)} files)
    """)

# ============= 使用示例 =============
if __name__ == "__main__":
    # 请修改以下参数为你自己的设置
    IMAGE_DIR = r"F:\Phenp Bench\PhenoBench-v110\aug-plant\images"           # 替换为你的原始图像路径
    MASK_DIR = r"F:\Phenp Bench\PhenoBench-v110\aug-plant\masks"             # 替换为你的原始掩码路径
    DATASET_NAME = "aug-plant"                    # 新数据集文件夹名
    OUTPUT_ROOT_PATH = r"E:\12_guorun\dataset\normal_datasets\seg"        # 指定输出根目录（会创建 agri_dataset 子文件夹）

    create_segmentation_dataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        dataset_name=DATASET_NAME,
        output_root_path=OUTPUT_ROOT_PATH,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )