import os

# 要处理的文件夹路径
folder = r"E:\12_guorun\dataset\normal_datasets\seg\aug-flood\test\masks"

for filename in os.listdir(folder):
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)

    # 如果文件名包含 "_lab"
    if name.endswith("_lab"):
        new_name = name.replace("_lab", "") + ext
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        # 重命名
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_name}")

print("✅ 处理完成！")
