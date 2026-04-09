import argparse
from pathlib import Path


def parse_list(value: str):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="D:\\GithubClones\\Segmentation\\Data\\WGAN_GP\\tomato", help="Folder containing images to rename")
    parser.add_argument("--prefix", type=str, default="tomato_WGAN_", help="Prefix to add to filenames")
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
                        help="Comma-separated extensions to include")
    parser.add_argument("--recursive", default=True, help="Process subfolders recursively")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions, no rename")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    exts = {e.lower() for e in parse_list(args.exts)}
    if args.recursive:
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        files = [p for p in root.iterdir() if p.is_file()]

    if exts:
        files = [p for p in files if p.suffix.lower() in exts]

    files = sorted(files, key=lambda p: p.name)
    if not files:
        print("No files matched.")
        return

    to_rename = []
    new_paths = set()
    for p in files:
        if p.name.startswith(args.prefix):
            continue
        new_name = args.prefix + p.name
        new_path = p.parent / new_name
        if new_path in new_paths:
            raise FileExistsError(f"Duplicate target path: {new_path}")
        if new_path.exists():
            raise FileExistsError(f"Target exists: {new_path}")
        new_paths.add(new_path)
        to_rename.append((p, new_name))

    if not to_rename:
        print("All matched files already have the prefix.")
        return

    print(f"Renaming {len(to_rename)} files in {root}")
    if args.dry_run:
        for p, new_name in to_rename[:10]:
            print(f"[Dry-run] {p.name} -> {new_name}")
        if len(to_rename) > 10:
            print(f"[Dry-run] ... and {len(to_rename) - 10} more")
        return

    tmp_pairs = []
    for idx, (p, new_name) in enumerate(to_rename, 1):
        tmp_name = f"__tmp__{idx}__{p.name}"
        tmp_path = p.parent / tmp_name
        if tmp_path.exists():
            raise FileExistsError(f"Temp exists: {tmp_path}")
        tmp_pairs.append((p, tmp_path, p.parent / new_name))

    for src, tmp, _ in tmp_pairs:
        src.rename(tmp)
    for _, tmp, dst in tmp_pairs:
        tmp.rename(dst)

    print("Done.")


if __name__ == "__main__":
    main()
