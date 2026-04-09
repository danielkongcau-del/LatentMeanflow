import os
from PIL import Image
import sys


def convert_png_to_jpg_and_remove(folder_path):
    """
    将指定文件夹下的所有PNG文件转换为JPG并删除原文件。

    Args:
        folder_path (str): 目标文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 警告：文件夹不存在，已跳过 - {folder_path}")
        return

    # 统计
    converted_count = 0
    failed_count = 0

    print(f"\n正在处理文件夹: {folder_path}")
    print("-" * 50)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            try:
                # 构造JPG文件名
                base_name = os.path.splitext(filename)[0]
                jpg_filename = base_name + '.jpg'
                jpg_file_path = os.path.join(folder_path, jpg_filename)

                # 打开图像
                with Image.open(file_path) as img:
                    # 处理透明度：转为RGB，白底
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        if img.mode in ('RGBA', 'LA'):
                            background.paste(img, mask=img.split()[-1])
                        rgb_img = background
                    else:
                        rgb_img = img.convert('RGB')

                    # 保存为JPG
                    rgb_img.save(jpg_file_path, 'JPEG', quality=95, optimize=True)

                # ✅ 保存成功后，删除原PNG文件
                os.remove(file_path)
                print(f"✅ 转换并删除: {filename} -> {jpg_filename}")
                converted_count += 1

            except Exception as e:
                print(f"❌ 转换失败 {filename}: {str(e)}")
                failed_count += 1

    print(f"\n📁 {os.path.basename(folder_path)} 处理完成:")
    print(f"   成功转换并删除: {converted_count} 个文件")
    print(f"   转换失败: {failed_count} 个文件")


def main():
    # 🔧 === 请修改为你的三个文件夹路径 ===
    folders_to_convert = [
        r"F:\Phenp Bench\PhenoBench-v110\aug-plant\images"
    ]
    # 🔧 ================================

    print("🔄 开始批量转换 PNG 为 JPG 并删除原文件...")
    print("=" * 60)

    for folder in folders_to_convert:
        clean_folder = folder.strip().strip('"\'')
        if clean_folder:
            if os.path.exists(clean_folder):
                convert_png_to_jpg_and_remove(clean_folder)
            else:
                print(f"⚠️  路径无效，已跳过: {clean_folder}")

    print("\n" + "=" * 60)
    print("🎉 所有文件夹处理完成！")
    print("🗑️  原PNG文件已全部删除，仅保留JPG文件。")


if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        print("❌ 错误：未安装Pillow库。")
        print("请运行：pip install Pillow")
        sys.exit(1)

    main()