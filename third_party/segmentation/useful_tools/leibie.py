import os
from PIL import Image
import numpy as np
from collections import Counter


def count_pixel_values(image_folder):
    # 初始化一个计数器来存储所有图像的像素值
    pixel_counter = Counter()

    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 构建完整的文件路径
        file_path = os.path.join(image_folder, filename)

        # 检查文件是否为图像文件
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        # 打开图像文件
        with Image.open(file_path) as img:
            # 将图像转换为 numpy 数组
            img_array = np.array(img)

            # 如果图像是多通道的（如 RGB），将其展平为一维数组
            if len(img_array.shape) > 2:
                # 将每个像素点的多个通道值组合成一个元组
                img_array = img_array.reshape(-1, img_array.shape[-1])
                pixels = map(tuple, img_array)
            else:
                # 对于单通道图像，直接展平
                pixels = img_array.flatten()

            # 更新计数器
            pixel_counter.update(pixels)

    return pixel_counter


def print_pixel_statistics(pixel_counter):
    # 打印统计结果
    print("像素值种类及数量：")
    for pixel, count in sorted(pixel_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"像素值: {pixel}, 数量: {count}")


if __name__ == "__main__":
    # 指定图像文件夹路径
    image_folder = r"E:\12_guorun\dataset\normal_datasets\seg\aug-plant\val\masks"

    # 统计像素值
    pixel_counter = count_pixel_values(image_folder)

    # 打印统计结果
    print_pixel_statistics(pixel_counter)