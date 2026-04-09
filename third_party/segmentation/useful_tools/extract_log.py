import json
import os

# 假设 JSON 文件名为 'metrics.json'
file_path = 'metrics.json'

json_file_path = os.path.join(file_path, 'val_log.json')

# 读取 JSON 文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 提取 miou 和 pa 值
miou_values = [entry['miou'] for entry in data]
pa_values = [entry['pa'] for entry in data]

# 将 miou 值保存到 miou.txt 文件中
with open(os.path.join(file_path, 'miou.txt'), 'w') as miou_file:
    for miou in miou_values:
        miou_file.write(f'{miou}\n')

# 将 pa 值保存到 pa.txt 文件中
with open(os.path.join(file_path, 'pa.txt'), 'w') as pa_file:
    for pa in pa_values:
        pa_file.write(f'{pa}\n')