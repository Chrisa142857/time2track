"""Population-average rest mesh from the training split, and a fixed validation subsample.

The template averages the frame-0 rest meshes of all TRAIN subjects (the seed-42 80/20 split of
training), each centred on its own centroid as in ukbiobank_dataset.py, so no validation subject
enters it. The subsample is drawn from the validation split with numpy RandomState(--subsample-seed).
Writes <out-dir>/cohort_mean_template.npy (388, 3), <out-dir>/val_subsample_<n>.txt (sorted subject
ids) and <out-dir>/prep_meta.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
from dataset.ukbiobank_dataset import UKbiobankHeartMesh  # noqa: E402


def frame0_centered(npy_root, mesh_d):
    arr = np.load(os.path.join(npy_root, mesh_d, "frames.npy"), mmap_mode="r")
    f0 = np.asarray(arr[0], dtype=np.float64)          # (N,3)
    return f0 - f0.mean(axis=0, keepdims=True)         # centre at own centroid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(SRC / "configs" / "config_ecg_heart_delaywavev3.yaml"),
                    help="model config; its dataset fields are relative to src/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-subsample", type=int, default=2000)
    ap.add_argument("--subsample-seed", type=int, default=123)
    args = ap.parse_args()
    config, out_dir = os.path.abspath(args.config), os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(SRC)  # config paths are relative to src/

    cfg = OmegaConf.load(config)
    OmegaConf.set_struct(cfg, False)
    npy_root = cfg.dataset.npy_dir

    train_ds = UKbiobankHeartMesh("train", cfg.dataset)
    val_ds = UKbiobankHeartMesh("val", cfg.dataset)
    print(f"train={len(train_ds.subjects)}  val={len(val_ds.subjects)}")

    # ---- cohort-mean template over TRAIN frame-0 rest meshes -----------------
    acc = None
    n = 0
    for sid, _ecg, mesh_d in train_ds.subjects:
        f0 = frame0_centered(npy_root, mesh_d)
        acc = f0 if acc is None else acc + f0
        n += 1
        if n % 5000 == 0:
            print(f"  accumulated {n} train subjects")
    template = (acc / n).astype(np.float32)             # (N,3)
    np.save(os.path.join(out_dir, "cohort_mean_template.npy"), template)
    tmpl_scale = float(np.sqrt((template.astype(np.float64) ** 2).sum(-1).mean()))
    print(f"template: shape={template.shape} n_train={n} rms_scale={tmpl_scale:.4f}")

    # ---- fixed val subsample -------------------------------------------------
    val_sids = [s[0] for s in val_ds.subjects]
    rng = np.random.RandomState(args.subsample_seed)
    k = min(args.n_subsample, len(val_sids))
    idx = rng.choice(len(val_sids), size=k, replace=False)
    sub = sorted(val_sids[i] for i in idx)
    with open(os.path.join(out_dir, f"val_subsample_{k}.txt"), "w") as f:
        f.write("\n".join(sub) + "\n")
    print(f"val subsample: {len(sub)} subjects (seed={args.subsample_seed})")

    meta = {
        "n_train": n, "n_val": len(val_sids),
        "template_rms_scale": tmpl_scale,
        "template_centroid": template.mean(0).tolist(),
        "n_subsample": len(sub), "subsample_seed": args.subsample_seed,
        "config": config,
    }
    with open(os.path.join(out_dir, "prep_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
