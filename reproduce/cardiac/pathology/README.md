# Pathology subgroups (Table 2)

Reproduces Table 2 of the camera-ready, "Diseased minus matched healthy subjects (n=981, lower is
better)", and the numbers of the paragraph "Pathology and Robustness" in Sec. 4 ("We have compared
the 981 validation subjects with a cardiac ICD-10 code with as many randomly sampled healthy
subjects. Table 2 shows that Time2Track has the smallest RMSE change and the only decrease of
|APCR−1|, which rises by 0.034 for CNN and 0.064 for TeCh."). The same subject lists are the input
of the robustness analysis (Tables 7 and 8).

| Script | What it does |
|---|---|
| `build_subgroups.py` | splits the validation subjects that have predictions into ICD-10 subgroups and a healthy pool |
| `subgroup_metrics.py` | RV, RMSE, PAR, APCR and APDR per subgroup and variant, with a subgroup bootstrap and a matched-n healthy bootstrap |
| `pathology_table.py` | prints Table 2 from the JSON of `subgroup_metrics.py` |

Subgroups are matched code by code against UK Biobank field 41270 (all ICD-10 diagnoses):

| List | ICD-10 | n in the paper run |
|---|---|---|
| `any_cardiac` | I20 to I25, I42 to I50 | 981 |
| `ischemic` | I20 to I25 | 577 |
| `arrhythmia_cond` | I44 to I49 | 542 |
| `atrial_fib` | I48 | 333 |
| `heart_failure` | I50 | 98 |
| `conduction_block` | I44, I45 | 170 |
| `cardiomyopathy` | I42, I43 | 30 |
| `healthy` | no code of `any_cardiac` | 7,112 |

## Inputs

1. A UK Biobank tabular extract (tab-separated, first column `eid`, a column `p41270` holding the
   pipe-separated ICD-10 codes of field 41270), obtained through an approved UK Biobank application.
2. Full validation predictions (8,093 subjects) of Time2Track, CNN and TeCh from the released
   weights (`bash download_ckpts.sh`), from the repository root:

   ```bash
   for c in config_ecg_heart_delaywavev3 config_ecg_heart config_ecg_heart_tech; do
     bash eval_ecg_heart.sh "" src/configs/$c.yaml -1
   done
   ```

   This writes `checkpoints/<variant>/npy_eval/{subid}_{pred,gt}.npy`.

## Commands

```bash
python reproduce/cardiac/pathology/build_subgroups.py --icd-tsv <ukb extract>.tsv --out <lists>
python reproduce/cardiac/pathology/subgroup_metrics.py --lists <lists> --out <out dir>/subgroup_metrics.json
python reproduce/cardiac/pathology/pathology_table.py <out dir>/subgroup_metrics.json
```

Defaults: `build_subgroups.py --pred-dir checkpoints/ecg_heart_delaywavev3_8L_256d_49frames/npy_eval`
(the evaluated subjects), `subgroup_metrics.py --root checkpoints --ndraws 1000 --seed 0`,
`--variants Ours=... CNN=... TeCh=...` (the first variant also supplies the GT files) and all seven
subgroups, `pathology_table.py --subgroup any_cardiac`. Relative defaults are resolved against the
repository root.

## Outputs

- `<lists>/<subgroup>.txt` and `<lists>/healthy.txt` (one subject id per line) and
  `<lists>/counts.json`. These files contain UK Biobank subject ids, so keep them out of the
  repository. The robustness analysis reads `<lists>/any_cardiac.txt` (Table 7, noise and lead
  dropout) and `<lists>/conduction_block.txt` (Table 8, τ_max).
- `subgroup_metrics.json`: per variant and subgroup the point estimate (`point`, order RV, RMSE, PAR,
  APCR, APDR) and `[median, 2.5 %, 97.5 %]` of the subgroup bootstrap (`boot_subgroup`) and of the
  matched-n healthy bootstrap (`boot_healthy_matched`).
- `pathology_table.py` prints, per variant, the subgroup bootstrap median minus the median of the
  variant's own matched healthy bootstrap, as |x−1| for PAR, APCR and APDR. These are the cells of
  Table 2. For the paper run they are:

  ```
  Delta         RMSE   |PAR-1|  |APCR-1|  |APDR-1|
  Ours        +0.001    +0.009    -0.025    +0.004
  CNN         +0.005    -0.007    +0.034    -0.001
  TeCh        +0.005    +0.008    +0.064    -0.005
  ```

The point estimates do not depend on random numbers. With seed 0 this script reproduced the
point estimates and bootstrap percentiles of the paper run bit for bit (checked for Ours on
`any_cardiac`, the first block of the random stream, on an H100). The bootstrap indices come from
`torch.randint` on a CUDA generator, whose stream is not guaranteed to be identical across PyTorch
versions and GPU setups. A second recorded run of the original code drew different indices and
differs from the paper run by at most 0.001 in the Table 2 cells (Ours |PAR−1| +0.008 and
|APCR−1| −0.026, TeCh |APCR−1| +0.063).

## Runtime

One GPU with about 12 GB of free memory (the per-subject statistics of 8,093 subjects are held in
float64 on the GPU). On an H100, the check above (GT, one variant and one subgroup of 981 subjects
with 2 x 1,000 draws) took 41 s, so the full run of three variants and seven subgroups takes minutes.
