import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.utils import ensure_segmentation_vendor_on_path


def _run(command):
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _safe_rmtree(path):
    path = Path(path).resolve()
    if path.exists():
        shutil.rmtree(path)


def _create_mock_teacher_run(run_dir, *, net_name="unet", out_channels=7, height=128, width=128):
    ensure_segmentation_vendor_on_path()
    from choices import choose_net

    run_dir.mkdir(parents=True, exist_ok=True)
    model = choose_net(net_name, out_channels=out_channels, img_size=height)
    checkpoint_path = run_dir / f"{net_name}_best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    train_args = {
        "net_name": net_name,
        "out_channels": int(out_channels),
        "height": int(height),
        "width": int(width),
        "batch_size": 1,
        "epoch": 0,
        "save_dir": str(run_dir),
    }
    (run_dir / "train_args.json").write_text(json.dumps(train_args, indent=2), encoding="utf-8")
    return checkpoint_path


def _build_mock_generated_root(prepared_dataset_root, generated_root, nfe_values):
    val_image_dir = Path(prepared_dataset_root) / "val" / "images"
    val_mask_dir = Path(prepared_dataset_root) / "val" / "masks"
    image_paths = sorted(val_image_dir.glob("*.png"))[:2]
    if not image_paths:
        image_paths = sorted(val_image_dir.iterdir())[:2]
    if len(image_paths) < 2:
        raise RuntimeError("Need at least two validation images for the segmentation teacher selfcheck.")
    for nfe in nfe_values:
        nfe_dir = generated_root / f"nfe{int(nfe)}"
        image_dir = nfe_dir / "generated_image"
        input_mask_dir = nfe_dir / "input_mask_raw"
        image_dir.mkdir(parents=True, exist_ok=True)
        input_mask_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            stem = image_path.stem
            shutil.copy2(image_path, image_dir / f"{stem}.png")
            shutil.copy2(val_mask_dir / f"{stem}.png", input_mask_dir / f"{stem}.png")


def main():
    base_dir = REPO_ROOT / "outputs" / "selfcheck_segmentation_teacher_workflow"
    prepared_root = base_dir / "prepared_teacher_data"
    mock_run_dir = base_dir / "mock_teacher_run_unet"
    bakeoff_outdir = base_dir / "bakeoff"
    generated_root = base_dir / "mock_generated_validation"
    teacher_mask_root = base_dir / "precomputed_teacher_masks" / "validation"
    layout_eval_outdir = base_dir / "layout_eval"

    _safe_rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "scripts/prepare_segmentation_teacher_data.py",
            "--src-root",
            "data/remote",
            "--dst-root",
            str(prepared_root),
            "--label-spec",
            "configs/label_specs/remote_semantic.yaml",
            "--splits",
            "train",
            "val",
            "test",
            "--overwrite",
        ]
    )

    _create_mock_teacher_run(mock_run_dir)
    _run(
        [
            sys.executable,
            "scripts/eval_segmentation_teacher_candidates.py",
            "--dataset-root",
            str(prepared_root),
            "--split",
            "val",
            "--label-spec",
            "configs/label_specs/remote_semantic.yaml",
            "--candidate-run",
            f"smoke_unet={mock_run_dir}",
            "--batch-size",
            "1",
            "--max-samples",
            "2",
            "--outdir",
            str(bakeoff_outdir),
            "--overwrite",
        ]
    )

    _build_mock_generated_root(prepared_root, generated_root, nfe_values=[8, 4])
    _run(
        [
            sys.executable,
            "scripts/export_teacher_masks.py",
            "--run-dir",
            str(mock_run_dir),
            "--generated-root",
            str(generated_root),
            "--split",
            "validation",
            "--batch-size",
            "1",
            "--nfe-values",
            "8",
            "4",
            "--outdir",
            str(teacher_mask_root),
            "--overwrite",
        ]
    )

    _run(
        [
            sys.executable,
            "scripts/eval_mask_layout_faithfulness.py",
            "--config",
            "configs/latent_alphaflow_mask2image_unet.yaml",
            "--generated-root",
            str(generated_root),
            "--outdir",
            str(layout_eval_outdir),
            "--split",
            "validation",
            "--n-samples",
            "2",
            "--batch-size",
            "1",
            "--nfe-values",
            "8",
            "4",
            "--teacher-mask-root",
            str(teacher_mask_root),
            "--skip-lpips",
            "--overwrite",
        ]
    )

    required_paths = [
        prepared_root / "dataset_manifest.json",
        bakeoff_outdir / "summary.json",
        teacher_mask_root / "nfe8" / "teacher_mask_raw",
        teacher_mask_root / "nfe4" / "teacher_mask_raw",
        layout_eval_outdir / "summary.json",
        layout_eval_outdir / "report.md",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Segmentation teacher selfcheck failed. Missing path(s): {missing}")

    bakeoff_summary = json.loads((bakeoff_outdir / "summary.json").read_text(encoding="utf-8"))
    if bakeoff_summary.get("winner") != "smoke_unet":
        raise AssertionError(f"Unexpected bakeoff winner: {bakeoff_summary.get('winner')}")

    print(f"Segmentation teacher workflow selfcheck passed under {base_dir}")


if __name__ == "__main__":
    main()
