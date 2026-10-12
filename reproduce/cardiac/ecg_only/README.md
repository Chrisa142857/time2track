# ECG as the only input (Table 3)

Reproduces the p̂0 column of Table 3 of the camera-ready, "Own (p0) vs. population (p̂0) rest
mesh", and the paragraph "ECG as the Only Input" in Sec. 4 ("Replacing the subject's rest mesh p0
with the population average p̂0 at inference, without retraining, leaves a resting 12-lead ECG as
the only input. Table 3 shows that the cost falls on fidelity (RV −0.046, RMSE +0.031) rather than
on diversity.").

The p0 column of Table 3 repeats the Ours row of Table 1, and Δ is p̂0 minus p0.

| File | What it does |
|---|---|
| `build_template.py` | population-average rest mesh p̂0 (mean of the centred frame-0 meshes of the 32,369 training subjects) and a fixed 2,000-subject validation subsample (seed 123) |
| `run_inference.sh` | Time2Track inference on the subsample with p̂0 in place of each subject's rest mesh, through `src/eval_ecg_mesh_e5.py --init_pc_override` |
| `rest_mesh_bootstrap.py` | RV, PAR, \|APCR−1\|, \|APDR−1\| and RMSE of those predictions with a 1,000-draw bootstrap (seed 42) |

The model and the ECG are unchanged and only the rest mesh fed to the model is replaced. The
template contains no validation subject.

## Inputs

- The UK Biobank ECG and mesh data in the layout of the repository README, with the dataset paths of
  `src/configs/config_ecg_heart_delaywavev3.yaml` (relative to `src/`).
- The released Time2Track weights (`bash download_ckpts.sh`):
  `nips26-physctrl-outputs/ecg_heart_delaywavev3_8L_256d_49frames/best_model`.
- `PhysioWave/` and its `ecg.pth` at the repository root (see the repository README).

## Commands

From the repository root, with `<work>` an output directory outside the repository:

```bash
# 1. template and subsample
python reproduce/cardiac/ecg_only/build_template.py --out-dir <work>

# 2. inference with the population rest mesh (4 round-robin chunks at batch size 16, one per GPU)
GPUS="0 1 2 3" bash reproduce/cardiac/ecg_only/run_inference.sh \
    <work>/val_subsample_2000.txt <work>/eval_root/cohort_mean/npy_eval \
    --init_pc_override <work>/cohort_mean_template.npy

# 3. metrics and bootstrap
python reproduce/cardiac/ecg_only/rest_mesh_bootstrap.py --root <work>/eval_root \
    --subject-list <work>/val_subsample_2000.txt --template <work>/cohort_mean_template.npy \
    --out <work>/rest_mesh_bootstrap.json
```

Settings of the paper run, all defaults of these scripts: DDIM with 50 steps, guidance scale 2.0,
batch size 16, sampling seed 0. The DDIM noise is drawn per batch, so the batch composition fixes
each subject's sample. `run_inference.sh` reproduces the partition of the paper run: the sorted
subsample split round-robin into 4 lists (`split -n r/4`), each evaluated by its own call. The
chunks may run on one GPU in sequence (`GPUS=0`) with the same result. Bootstrap defaults:
`--B 1000 --seed 42 --n-xsubj 128 --conditions cohort_mean`. `--mesh-dir` (the
`{subid}_*/frames.npy` directory that supplies each subject's own rest mesh) defaults to
`dataset.npy_dir` of `--config`.

A second condition with the subject's own rest mesh (`run_inference.sh` without
`--init_pc_override`, into `<work>/eval_root/subject_specific/npy_eval`, then
`--conditions cohort_mean subject_specific`) gives the same-pipeline reference on the same draws.
It does not change the cohort-mean numbers, because the bootstrap draws do not depend on the
conditions.

## Outputs

- `<work>/cohort_mean_template.npy` (388 x 3), `<work>/val_subsample_2000.txt` (UK Biobank subject
  ids, keep it out of the repository), `<work>/prep_meta.json`.
- `<work>/eval_root/cohort_mean/npy_eval/{subid}_{pred,gt}.npy`.
- `rest_mesh_bootstrap.json`: per condition the point estimate (`point`) and the median and 2.5 %
  and 97.5 % percentiles over the draws (`ci`).

The p̂0 column of Table 3 is `ci.cohort_mean.<metric>.median`:

| Table 3 row | key | paper run |
|---|---|---|
| RV | `RV` | 0.5908 |
| PAR | `PAR` | 0.8673 |
| \|APCR−1\| | `APCR` | 0.0517 |
| \|APDR−1\| | `APDR` | 0.0362 |
| RMSE | `RMSE_offrem` | 0.1822 |

## Runtime

Inference: about 22 minutes per 500-subject chunk on one H100, so about 25 minutes with four GPUs
or 1.5 hours on one. The bootstrap of one condition took 14 s on one H100, and it reproduced the
numbers above bit for bit from the predictions of the paper run.
