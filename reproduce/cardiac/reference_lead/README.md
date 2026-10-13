# Reference lead (Appendix A.7)

Reproduces Appendix A.7 "Reference Lead" of the camera-ready paper:

- Table 9, "Sensitivity to the reference lead." (rows aVR and aVL)
- "In a preliminary ES timing classification without a reference lead, aggregating the full delay
  map by its mean or median reaches an AUC of 0.590 or 0.563, against 0.597 for the lead II delay."
- "The median inter-lead T-peak delay in UKB is 8 ms (interquartile range 2 to 30 ms)"

| paper number | produced by |
|---|---|
| Table 9, aVR and aVL rows | `configs/ref_aVR.yaml`, `configs/ref_aVL.yaml` with `train_ecg_heart.sh` and `eval_ecg_heart.sh`, then `score_arms.py` |
| Table 9, lead II row | not produced here: it is the Ours row of Table 1 |
| AUC 0.597 (lead II), 0.590 (mean), 0.563 (median) | `extract_delay_es.py`, then `reference_free_auc.py` |
| 8 ms (IQR 2 to 30 ms) | `tpeak_delay_stats.py` |

Common inputs: the trained weights from `bash download_ckpts.sh` (repository root), which land in
`checkpoints/ecg_heart_delaywavev3_8L_256d_49frames/best_model`, the UK Biobank data at
the dataset paths of `src/configs/config_ecg_heart_delaywavev3.yaml` (see the main README), and
the unperturbed validation evaluation of the trained model:

```bash
bash eval_ecg_heart.sh "" src/configs/config_ecg_heart_delaywavev3.yaml -1
```

which writes `checkpoints/ecg_heart_delaywavev3_8L_256d_49frames/npy_eval/`. All
commands below run from the repository root.

## Table 9: retraining with aVR or aVL as the reference lead

Each arm fine-tunes the trained Time2Track model with a different reference lead `r` of the delay
branch, `ref_lead: 3` (aVR) or `ref_lead: 4` (aVL) instead of `1` (lead II). The configs are copies
of `src/configs/config_ecg_heart_delaywavev3.yaml` with only these changes:

| field | base | arm |
|---|---|---|
| `model_config.delaywavev3_encoder.ref_lead` | 1 | 3 (aVR) or 4 (aVL) |
| `warm_start_from` | absent | the trained `best_model` (weights only, fresh optimizer and LR schedule) |
| `max_train_steps` / `num_train_epochs` | null / 200 | 12000 / 10000 (the step budget decides) |
| `early_stop_patience` | 3 | 0 (no early stop, equal budget for every arm) |
| `validation_steps` / `checkpoints_total_limit` | 500 / 5 | 250 / 3 |
| `output_dir` | `../checkpoints/ecg_heart_delaywavev3_...` | `../checkpoints/reference_lead_<lead>` |

Everything else is the base recipe: seed 42, one GPU with batch 14 and gradient accumulation 4
(56 samples per step, about 21 epochs in 12,000 steps), learning rate 1e-5 with cosine schedule and
200 warmup steps, bf16, the same losses, data and split. The scored checkpoint is the final
`checkpoint-12000`, not `best_model`, so every arm has the same budget and no validation selection.

```bash
for lead in aVR aVL; do
  CUDA_VISIBLE_DEVICES=0 bash train_ecg_heart.sh 1 reproduce/cardiac/reference_lead/configs/ref_${lead}.yaml
  bash eval_ecg_heart.sh checkpoints/reference_lead_${lead}/checkpoint-12000 \
       reproduce/cardiac/reference_lead/configs/ref_${lead}.yaml -1
done
python reproduce/cardiac/reference_lead/score_arms.py \
  --dir aVR=checkpoints/reference_lead_aVR/npy_eval \
  --dir aVL=checkpoints/reference_lead_aVL/npy_eval \
  --out <out dir>/reference_lead_scores.json
```

`eval_ecg_heart.sh` writes `npy_eval/` next to the checkpoint (DDIM 50 steps, CFG 2.0, sample seed
0, all 8,093 validation subjects). `score_arms.py` scores every dir on the subjects common to all
dirs: RV, RMSE and PAR over all subjects and APCR and APDR over the first 128 sorted subjects, the
convention of Table 1. The `POINT` lines and the `point` entry of the JSON are the cells of Table 9
(aVR 0.590 / 0.181 / 0.948 / 0.041 / 0.022, aVL 0.592 / 0.181 / 0.934 / 0.074 / 0.024 in the paper
run). The JSON also holds 95 % intervals from a 1,000-draw paired subject bootstrap (seed 0), and
with `--ref <label>` the paired differences against that arm. Neither appears in the paper.

The lead II row of Table 9 is the fully trained model of Table 1, not a 12,000-step fine-tune. A
like-for-like lead II arm is the same config with `ref_lead: 1` and its own `output_dir`.

Runtime on one NVIDIA H100 NVL: about 8 h of training and about 5.5 h of evaluation per arm. The
scoring of two arms needs about 16 GB of GPU memory and a few minutes.

## ES timing classification without a reference lead

`extract_delay_es.py` computes, for every validation subject (the `{subid}_gt.npy` files of the
evaluation above, in sorted order):

- the pairwise lag tensor `D[r, c, s]` of lead `c` against each of the 12 reference leads `r` at
  the 3 wavelet scales, with the trained wavelet front-end, FFT cross-correlation and soft-argmax of
  the delay branch, on the CPU,
- the ground-truth ES frame, the frame of minimum convex-hull volume of the GT trajectory within
  frames 8 to 32 (256 to 1024 ms) of the 50-frame, 32 ms grid.

`reference_free_auc.py` builds three per-lead timing vectors from `D`: the lead II row (Time2Track),
and the mean and the median over all other reference leads, centred across leads (no reference
lead). Each is fed to a standardised L2 logistic regression that separates early from late ES (outer
tertiles of the ES frame, n = 6,880), and the score is the ROC-AUC of the out-of-fold predictions of
a 5-fold stratified CV with `random_state` 42.

```bash
python reproduce/cardiac/reference_lead/extract_delay_es.py --out <out dir>/delay_es.npz
python reproduce/cardiac/reference_lead/reference_free_auc.py --features <out dir>/delay_es.npz
```

Defaults are derived from `src/configs/config_ecg_heart_delaywavev3.yaml`: the checkpoint is
`<output_dir>/best_model`, the GT dir `<output_dir>/npy_eval` and the ECG dir `dataset.ecg_dir`
(override with `--checkpoint`, `--npy-eval`, `--ecg-dir`). The second step prints
`leadII AUC 0.5972`, `mean AUC 0.5900`, `median AUC 0.5627` for the paper run. Both steps run on
the CPU in a few minutes. The subject order fixes the CV folds, so keep the sorted order.

## Size of the inter-lead T-peak delays

```bash
python reproduce/cardiac/reference_lead/tpeak_delay_stats.py --delay-matrix ukb_heart_mesh_visual/all_ecg_delay_matrix.npz
```

`all_ecg_delay_matrix.npz` holds, per subject, the 12×12 matrix of inter-lead T-peak delays in ms
from the NeuroKit2 delineation of the resting ECG that underlies Fig. 1b (keys `filename`,
`T_peak`). It is built by `data_prep/ecg_delay_matrix.py` (step 6 of `data_prep/README.md`). The script pools |delay| over the 66 lead pairs and the 8,093 validation subjects (the
`{subid}_gt.npy` files of the evaluation above, or `--npy-eval`) and prints, for the paper run,
`median 8 ms, IQR 2 to 30 ms, p95 186 ms, max 434 ms`.
