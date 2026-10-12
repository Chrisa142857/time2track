# Robustness to electrode noise, missing leads and τ_max (Appendix A.6)

Reproduces Appendix A.6 "Robustness to Electrode Noise, Missing Leads and τmax" of the camera-ready
paper:

- Table 7, "Electrode noise and lead dropouts at inference on the diseased subgroup."
- Table 8, "Sensitivity to τmax at inference, default in bold (95% bootstrap intervals for RV and RMSE)."

The trained Time2Track model (`ecg_heart_delaywavev3_8L_256d_49frames`) is run on the validation
subjects of a subgroup with one perturbation at inference time, and the five whole-heart metrics
are computed against the unperturbed GT with a subject bootstrap.

- Electrode noise: per-lead Gaussian noise with σ set from the power of each lead to reach 20, 10
  or 5 dB SNR. One fixed noise pattern (seed 1234) is rescaled per SNR.
- Lead dropout: the channels of lead II, of V2, or of V1, V2 and V3 are set to zero.
- τ_max: the search window of the delay branch is set to 25, 75 or 100 samples (±50, ±150,
  ±200 ms at 500 Hz) instead of the trained 50 samples. The ECG is unchanged.

Every condition uses the same subject order and batches (batch 32, `shuffle=False`) and the same
seed-0 initial noise per batch, so clean versus perturbed is a within-subject comparison.

## Files

| file | role |
|---|---|
| `perturbed_inference.py` | inference under one perturbation, writes `{subid}_pred.npy` (skip-if-exists) |
| `robustness_metrics.py` | RV, RMSE, PAR, APCR, APDR and a 1,000-draw subject bootstrap per condition |
| `run_inference.sh` | the 11 conditions of Tables 7 and 8 |
| `run_metrics.sh` | the two metric calls that give Table 7 and Table 8 |

The metric primitives are imported from `reproduce/metrics/cardiac_stats.py`, and the model,
dataset and DDIM pipeline from `src/`.

## Inputs

1. Trained weights: `bash download_ckpts.sh` (repository root) extracts them to
   `nips26-physctrl-outputs/ecg_heart_delaywavev3_8L_256d_49frames/best_model`.
2. The UK Biobank ECG and mesh data at the dataset paths of
   `src/configs/config_ecg_heart_delaywavev3.yaml` (see the main README).
3. The unperturbed validation evaluation, which supplies the GT `{subid}_gt.npy`:

   ```bash
   bash eval_ecg_heart.sh "" src/configs/config_ecg_heart_delaywavev3.yaml -1
   ```

   This writes `nips26-physctrl-outputs/ecg_heart_delaywavev3_8L_256d_49frames/npy_eval/`.
4. Subgroup lists from `reproduce/cardiac/pathology/build_subgroups.py`, which writes
   `<lists dir>/<subgroup>.txt` (one subject id per line). Two are used:
   `any_cardiac.txt`, the diseased subgroup of Table 7 (ICD-10 I20 to I25 and I42 to I50, 981
   validation subjects in the paper run), and `conduction_block.txt` for Table 8 (ICD-10 I44 to
   I45, 170 validation subjects).

## Commands

From the repository root, with the environment of the main README:

```bash
CUDA_VISIBLE_DEVICES=0 bash reproduce/cardiac/robustness/run_inference.sh <lists dir> <out dir>
CUDA_VISIBLE_DEVICES=0 bash reproduce/cardiac/robustness/run_metrics.sh   <lists dir> <out dir>
```

`run_inference.sh` also accepts a subset of condition names after `<out dir>`, so the sweep can be
split over GPUs. Settings of the paper run, all defaults of the scripts: DDIM with 50 steps, CFG
scale 2.0, batch 32, noise seed 1234, sampling seed 0 per batch, 1,000 bootstrap draws with seed 0.
`CONFIG`, `CKPT`, `GT_DIR` and `PYTHON` can be set in the environment. By default the checkpoint
and the GT dir are `best_model` and `npy_eval` under the `output_dir` of the config, as in
`eval_ecg_heart.sh`.

| condition | subgroup | perturbation | table row |
|---|---|---|---|
| `ac_clean` | any_cardiac | none | Table 7, clean |
| `ac_noise20`, `ac_noise10`, `ac_noise5` | any_cardiac | `--mode noise --snr 20/10/5` | Table 7, noise |
| `ac_drop_V2` | any_cardiac | `--mode dropout --drop-leads 7` | Table 7, drop V2 |
| `ac_drop_II` | any_cardiac | `--mode dropout --drop-leads 1` | Table 7, drop II |
| `ac_drop_chest3` | any_cardiac | `--mode dropout --drop-leads 6,7,8` | Table 7, drop V1, V2, V3 |
| `cb_clean` | conduction_block | none | Table 8, τ_max 50 (bold) |
| `cb_tau25`, `cb_tau75`, `cb_tau100` | conduction_block | `--mode taumax --tau-max 25/75/100` | Table 8 |

Lead indices are 0-based in the order I, II, III, aVR, aVL, aVF, V1 to V6.

## Outputs

`<out dir>/<condition>/npy_eval/{subid}_pred.npy` from the inference, then
`<out dir>/table7_noise_dropout.json` and `<out dir>/table8_taumax.json`, each with `point` (the
five metrics over all subjects) and `boot` (per metric [median, 2.5 %, 97.5 %] over the draws). The
table cells are the bootstrap medians: RV, RMSE and PAR as they are, and |median APCR − 1| and
|median APDR − 1|. The bracketed intervals of Table 8 are the 2.5 % and 97.5 % values of RV and RMSE.
`robustness_metrics.py` prints one `table row` line per condition in that form. The bootstrap
generator is consumed in condition order, so the order in `run_metrics.sh` is part of the result.
Applied to the predictions of the paper run, `robustness_metrics.py` gives the numbers of Tables 7
and 8.

## Runtime

On one NVIDIA H100 NVL, about 40 min per any_cardiac condition (981 subjects) and about 10 min per
conduction_block condition (170 subjects), so about 5.5 h for the 11 conditions on one GPU. The
metrics of Table 8 take about a minute on one GPU. Table 7 (981 subjects, 7 conditions, 1,000
draws each) takes longer, on the order of half an hour. Neither runtime was recorded in the paper
run.
