# Error bars of the diversity metrics (Table 1, Appendix A.5)

Reproduces the `±` of the |APCR−1| and |APDR−1| columns of Table 1 and the ranges stated in
Appendix A.5 "Error Bars of the Diversity Metrics".

- Table 1 caption: "Whole-heart 3D vertex trajectory metrics (Pred vs. GT). Inter-subject diversity
  columns are reported as Pred/GT ratios, with ± the SD over the 10,000 paired subject bootstrap
  resamples described in Appendix A.5."
- Appendix A.5: "The ± of |APCR−1| and |APDR−1| in Table 1 is the SD of |APCR−1| and |APDR−1| across
  the resamples, which ranges from 0.037 to 0.078 and from 0.006 to 0.012, respectively."

`diversity_error_bars.py` draws 10,000 resamples (seed 0) of the diversity sample of the paper, the
128 validation subjects with the smallest subject ids, with replacement. The numerator and the
denominator of APCR and APDR are recomputed on the same resample, and one resample matrix is shared
by all seven methods. The `±` is the SD of |APCR−1| and |APDR−1| across the resamples
(`se_absAPCRm1`, `se_absAPDRm1` in the output). The metric primitives are imported from
`reproduce/metrics/paper_metrics.py`.

The script first recomputes `xsubjPredRV`, `xsubjGTRV`, `xsDistPred` and `xsDistGT` of every method
and stops unless they match the metrics CSV of `eval_vertex_multiscale.py` to 5e-4.

The 95% bootstrap intervals of Table 8 (RV and RMSE under a change of τ_max) are not produced here.
They come from the robustness analysis.

## Inputs

1. Trained weights: `bash download_ckpts.sh` (repository root) extracts every variant to
   `nips26-physctrl-outputs/<variant>/best_model`.
2. Validation predictions of the seven Table 1 encoders, from the repository root:

   ```bash
   for c in config_ecg_heart config_ecg_heart_tech config_ecg_heart_physiowave \
            config_ecg_heart_shift_invariant config_ecg_heart_delay config_ecg_heart_delay_qrs \
            config_ecg_heart_delaywavev3; do
     bash eval_ecg_heart.sh "" src/configs/$c.yaml -1
   done
   ```

   This writes `nips26-physctrl-outputs/<variant>/npy_eval/{subid}_{pred,gt}.npy` (sample seed 0).
3. The metrics CSV, used by the reproduction check:

   ```bash
   python eval_vertex_multiscale.py --root nips26-physctrl-outputs --mesh-info <mesh asset dir> --out outputs
   ```

## Command

```bash
python reproduce/cardiac/error_bars/diversity_error_bars.py --out <out dir>/diversity_error_bars.json
```

Defaults: `--root nips26-physctrl-outputs`, `--csv outputs/vertex_multiscale/metrics-avgstd_per_scale.csv`
(both relative to the repository root), `--nsamp 128`, `--B 10000`, `--seed 0`, and `--methods` the
seven rows of Table 1 as `Label=variant_dir`.

## Output

One line per method on stdout and a JSON with, per method, `APCR`, `APDR`, `absAPCRm1`, `absAPDRm1`
(full-precision point estimates on the 128 subjects), `se_absAPCRm1`, `se_absAPDRm1` (the `±` of
Table 1) and `csv_dev` (the deviation from the CSV). Values of the run behind the paper:

| Method | ± \|APCR−1\| | ± \|APDR−1\| |
|---|---|---|
| CNN | 0.058 | 0.007 |
| TeCh | 0.065 | 0.012 |
| PhysioWave | 0.037 | 0.006 |
| ShiftInvariant | 0.078 | 0.009 |
| Delay-T | 0.046 | 0.008 |
| Delay-QRS | 0.049 | 0.006 |
| Ours | 0.042 | 0.008 |

The bootstrap needs only the 128 subjects per method and runs on one GPU or on the CPU. On one H100
it finished in a few minutes and reproduced the table above bit for bit.
