import os
from pathlib import Path

# =================== 配置区 ===================
FOLDER_PATH = r"F:\新数据集\aug-fruit(a1p2)"        # <-- 修改这里
KEYWORDS = ["Pomegranate"]   # <-- 修改或添加关键字
# =============================================

def delete_files_with_keywords(folder_path, keywords):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ 路径不存在: {folder_path}")
        return

    deleted_count = 0
    for file_path in folder.rglob('*'):  # 递归遍历所有文件和文件夹
        if file_path.is_file():
            filename = file_path.name
            # 检查文件名是否包含任意一个关键词
            if any(keyword in filename for keyword in keywords):
                try:
                    file_path.unlink()  # 删除文件
                    print(f"✅ 已删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败: {file_path} | 错误: {e}")

    print(f"\n🎉 完成！共删除 {deleted_count} 个文件。")

# 运行函数
if __name__ == "__main__":
    print(f"🔍 正在扫描路径: {FOLDER_PATH}")
    print(f"📌 关键词: {KEYWORDS}")
    confirm = input("\n⚠️  确认要删除符合条件的文件吗？(y/N): ")
    if confirm.lower().strip() in ['y', 'yes', '是']:
        delete_files_with_keywords(FOLDER_PATH, KEYWORDS)
    else:
        print("👋 已取消操作。")