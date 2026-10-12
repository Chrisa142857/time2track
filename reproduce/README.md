# Reproducing the results beyond Table 1

The common pipeline comes first: build the inputs with `data_prep/`, train the models or fetch them with
`download_ckpts.sh`, run `eval_ecg_heart.sh` (it writes the per-subject predictions to `npy_eval/`), and
compute the whole-heart metrics of Table 1 and Figs. 4 to 6 with `eval_vertex_multiscale.py` and
`eval_facearea_multiscale.py` (see the top-level README). Each folder below then reproduces one further
result of the paper and has its own README with inputs, commands, settings and outputs.

```
reproduce/
├── metrics/              metric code shared by the folders below
├── cardiac/              UK Biobank ECG to 4D heart motion
│   ├── error_bars/       ± of the diversity columns of Table 1, Appendix A.5
│   ├── pathology/        Table 2, paragraph "Pathology and Robustness"
│   ├── ecg_only/         Table 3, paragraph "ECG as the Only Input" (population-average rest mesh)
│   ├── robustness/       Appendix A.6, Tables 7 and 8 (electrode noise, missing leads, τmax)
│   └── reference_lead/   Appendix A.7, Table 9 and the reference-free AUCs
└── semg/                 emg2pose sEMG to hand pose
    ├── joint_delay/      Fig. 1d (inter-joint delay of hand pose by gesture)
    └── delay_tokens/     Fig. 7, Appendix A.8 (delay tokens in vemg2pose and EMGFormer)
```

Fig. 1b (inter-lead T-wave delay) and the input of the Delay-T and Delay-QRS baselines come from
`../data_prep/ecg_delay_matrix.py`, then `../preliminary_ecg.ipynb`.

`metrics/cardiac_stats.py` computes the whole-heart metrics from per-subject statistics, so that subjects
can be bootstrap-resampled cheaply. `metrics/paper_metrics.py` computes the same metric family on generic
trajectories and serves the sEMG joint angles and the error bars.

`cardiac/pathology/` writes the subject lists that `cardiac/robustness/` reads (`any_cardiac.txt` for
Table 7, `conduction_block.txt` for Table 8), so run it first.
