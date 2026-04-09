import os
import json
import base64
import numpy as np
from PIL import Image
from labelme import utils

def json_to_png(json_file, output_dir):
    """
    将单个 Labelme JSON 文件转换为 PNG 标签图
    """
    # 读取 JSON 数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像数据
    imageData = data.get('imageData')
    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')

    # 转换为 numpy 数组
    img = utils.img_b64_to_arr(imageData)

    # 构建标签名到ID的映射（背景=0, animal=1）
    label_name_to_value = {'_background_': 0, 'fruit': 2}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name not in label_name_to_value:
            label_name_to_value[label_name] = len(label_name_to_value)

    shapes = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon' or shape['shape_type'] == 'rectangle':
            shapes.append(shape)
        else:
            print(f"⚠️ Skipping unsupported shape type: {shape['shape_type']}")

    # 生成标签图
    lbl, _ = utils.shapes_to_label(
        img.shape, shapes, label_name_to_value
    )

    # 使用PIL直接从numpy数组创建单通道图像并保存
    lbl_pil = Image.fromarray(lbl.astype(np.uint8), 'L') # 'L' 模式代表8位像素，黑白
    png_filename = os.path.splitext(os.path.basename(json_file))[0] + '.png'
    png_path = os.path.join(output_dir, png_filename)
    lbl_pil.save(png_path)

    # 保存类别映射文件 label_names.txt
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    with open(os.path.join(output_dir, 'label_names.txt'), 'w', encoding='utf-8') as f:
        for name in label_names:
            if name is not None:  # 确保只写入非空标签名
                f.write(name + '\n')

    print(f"✅ Saved: {png_path}")
    return lbl, label_names

def batch_convert(input_dir, output_dir):
    """
    批量转换指定文件夹内所有 .json 文件
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print("❌ No JSON files found in input directory.")
        return

    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        json_to_png(json_path, output_dir)

    print(f"🎉 Batch conversion completed. Output saved to: {output_dir}")


if __name__ == "__main__":
    # 直接在这里设置输入和输出目录
    input_directory = r'F:\新数据集\Origin data\pomegranate-03\jsons'  # 修改此路径为你存放JSON文件的文件夹
    output_directory = r'F:\新数据集\Origin data\pomegranate-03\masks-2'      # 修改此路径为你希望保存PNG文件的文件夹

    batch_convert(input_directory, output_directory)