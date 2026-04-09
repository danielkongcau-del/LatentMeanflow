import os

def delete_checkpoints(root_dir):
    # 需要匹配的关键字
    patterns = ["!.pt", "_20.pt", "_40.pt","_100.pt"]

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(p in filename for p in patterns):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败: {file_path}, 错误: {e}")

if __name__ == "__main__":
    # 修改为你要清理的根目录
    root_folder = r"E:\12_guorun\1-cv-classify\guoruns-segmentation-hub\work_dir"
    delete_checkpoints(root_folder)
