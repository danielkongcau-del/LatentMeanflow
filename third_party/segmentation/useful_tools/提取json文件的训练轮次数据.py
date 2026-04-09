import pandas as pd
import json


# 加载JSON数据
def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)


# 转换为DataFrame并保存为CSV
def json_to_csv(json_data, output_path):
    # 提取数据并构建DataFrame
    data = [{
        "epoch": entry["epoch"],
        "loss": entry["loss"],
        "miou": entry["miou"],
        "pa": entry["pa"]
    } for entry in json_data]

    df = pd.DataFrame(data)

    # 输出到CSV文件
    df.to_csv(output_path, index=False)
    print(f"数据已成功保存到 {output_path}")


# 使用示例
json_path = r'E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\work_dir\seg\pest\aug-swin_unet-h512w512-erode0-weighting_none\val_log.json'  # 替换为您的JSON文件路径
output_path = r'E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\useful_tools\1\output_file.csv'  # 替换为您想要保存的路径

# 载入JSON数据并保存为CSV
json_data = load_json(json_path)
json_to_csv(json_data, output_path)
