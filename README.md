# Official code for Time2Track

**From Heartbeat to Cardiochoreography: A Mechanics Foundation Model for Individualized 4D Cardiac Motion Generation Conditioned on Electrophysiology**

Ziquan Wei, Tingting Dan, Guorong Wu. NeurIPS 2026.

Project page: https://chrisa142857.github.io/time2track/ · Paper: https://chrisa142857.github.io/time2track/paper/time2track_neurips2026.pdf

## How to use

Generate the 4D motion of a heart from a resting 12-lead ECG, with or without a 3D mesh of that heart.

**1. Install.** From the repository root:

```bash
uv sync
git clone https://github.com/ForeverBlue816/PhysioWave && git -C PhysioWave checkout 6f7bec7
```

**2. Download the trained weights** (6.4 GB, extracted to `checkpoints/`):

```bash
bash download_ckpts.sh
```

**3. Generate.** A GPU is used when available (about 5 s of sampling per ECG on one H100, after the model loads).

```bash
# ECG only: the rest mesh is the population-average heart (assets/population_rest_mesh.npy)
python generate_4d_heart.py --ecg my_ecg.xml --out results/

# ECG + your own rest mesh (388-point control mesh at end-diastole, .ply or .npy)
python generate_4d_heart.py --ecg my_ecg.csv --fs 500 --rest-mesh my_heart.ply --out results/

# many subjects: a folder of ECGs and, optionally, a folder of meshes named like the ECGs
python generate_4d_heart.py --ecg ecgs/ --rest-mesh meshes/ --out results/
```

- **Two modes.** With `--rest-mesh`, the motion starts from your own heart shape. Without it, the
  motion starts from the population-average shape of the 32,369 UK Biobank training subjects, so the
  generated motion follows the ECG while the heart shape is the population average.
- **ECG input.** A GE MUSE XML as exported by UK Biobank (field 20205), or a 12 x T array (`.npy`,
  `.csv`, `.txt`) in the lead order I, II, III, aVR, aVL, aVF, V1 to V6, with its sampling rate in
  `--fs`. A CSV with lead names in its header is reordered by name. A recording of at most 1.4 s is
  treated as a median beat. A longer recording is treated as a rhythm strip, and its R-aligned beats
  are combined into a median beat. `--ecg-kind median|rhythm` overrides this choice, and
  `--ecg-kind ready` accepts an input that is already preprocessed (12 x 600), such as the output of
  `data_prep/`.
- **ECG preprocessing.** The script applies the preprocessing of training: resampling to 500 Hz, 600
  samples, the R-peak at sample 0 and z-normalization per lead. The model was trained on the 500 Hz
  median beats of UK Biobank, so the median beat of an ECG device is the input closest to the
  training data.
- **Rest mesh input.** The end-diastolic control mesh of the biv-volumetric-meshing pipeline: 388
  vertices in millimetres, in the coordinate frame of the UK Biobank meshes of Ugurlu et al. Use the
  PLY format of `data_prep/mesh_to_ply.py`, or a `.npy` of shape (388, 3). The mesh is centred on its
  centroid, as in training.
- **Outputs.** For each ECG, `<name>_motion.npy` holds 50 x 388 x 3 coordinates: the rest mesh,
  then the 49 generated frames of one cardiac cycle. `<name>.gif` animates the cycle,
  `<name>_ecg_input.npy` holds the preprocessed ECG, `--ply` adds one PLY mesh per frame, and
  `generation_config.json` records every setting of the run.
- **Model and sampling.** `--config` selects the model, by default `config_ecg_heart_delaywavev3.yaml`,
  whose weights `download_ckpts.sh` places under `checkpoints/` (`--checkpoint` points to other
  weights). Sampling uses 50 DDIM steps and a guidance scale of 2.0, as in the paper. The output is a
  sample of a diffusion model, so it changes with `--seed`, and also with `--batch-size` because the
  noise is drawn per batch. The default batch size of 1 keeps every subject independent of the others.
  Running several seeds shows the range of plausible motions for one ECG.

## Setup for training and evaluation

```
uv sync
```

The example GIFs in `example_ecg_gif/` are stored with Git LFS: run `git lfs install` before cloning,
or `git lfs pull` afterwards, to fetch the GIFs.

Two encoders are imported from directories at the repository root that are not part of this
repository. Clone them there before using the `tech`, `delay`, `physiowave` or `delaywave*` encoders:

| Directory | Source | Commit used | Needed by |
|---|---|---|---|
| `TeCh/` | https://github.com/Levi-Ackman/TeCh | `9a378cc` | `tech`, `delay` (Delay-T, Delay-QRS) |
| `PhysioWave/` | https://github.com/ForeverBlue816/PhysioWave | `6f7bec7` | `physiowave`, `delaywave*` |

The `physiowave` and `delaywave*` configs read the pretrained PhysioWave ECG weights from
`pretrained_path` (default `../PhysioWave/ecg.pth`, relative to `src/`). Download `ecg.pth` from the
"Pretrained Models" link of the PhysioWave README and put it in `PhysioWave/`. Every path field of a
config (`output_dir`, `pretrained_path`, `dataset.ecg_dir`, `dataset.npy_dir`, `dataset.mesh_dir`)
is relative to `src/`.

## Data

The experiments use UK Biobank data under Application ID 91673. UK Biobank data are available to
approved researchers at https://www.ukbiobank.ac.uk, so no subject data are distributed here.

- **ECG.** Resting 12-lead ECG (field 20205) at 500 Hz, 600 samples per subject, z-normalized per
  lead.
- **4D heart meshes.** 388 control points over 50 frames per cardiac cycle (frame 0 is the rest
  mesh), from the UK Biobank cardiac MR digital twins of Ugurlu et al., *PLOS ONE* 20(7): e0327158,
  2025.

`src/dataset/ukbiobank_dataset.py` reads either layout below. Subjects with both an ECG and a mesh
are paired, subjects whose displacement from the rest mesh exceeds `dataset.filter_max_displacement`
in more than `dataset.filter_max_noisy_frac` of their frames are dropped, and the remaining subjects
are split into train and validation by `dataset.train_split` (0.8) with a fixed permutation (seed 42).

```
{dataset.ecg_dir}/{subid}.npy                  (12, 600) float32, z-normalized per lead
{dataset.ecg_dir}/{subid}_2_0.csv              fallback: raw (12, 600), z-normalized on load
{dataset.npy_dir}/{subid}_{...}/frames.npy     (50, 388, 3) float32 (dataset.mesh_dir if npy_dir is unset)
{dataset.mesh_dir}/{subid}_{...}/frame_000.ply fallback: one ASCII PLY per frame
```

`data_prep/` builds this layout from the UK Biobank exports, together with the mesh template assets
read by the metric scripts (`control_materials.npy`, `control_face.ply`, passed as `--mesh-info` or
`--material_npy`) and the inter-lead delay matrix of the Delay-T and Delay-QRS baselines
(`all_ecg_delay_matrix.npz`). See `data_prep/README.md` for the steps in order.

## Pretrained weights

```bash
bash download_ckpts.sh
```

This downloads one 6.4 GB archive with the trained weights, config and training logs of Time2Track
and of every baseline and ablation, and extracts it to `checkpoints/<variant>/`, which is
the `output_dir` of the matching config. Evaluation and visualization then find
`<variant>/best_model` when the checkpoint argument is empty.

## Training and evaluation

**Train.** `NUM_GPUS` selects `src/configs/acc/<NUM_GPUS>gpu.yaml` (1, 2, 4, 7 or 8 GPUs), and the GPU
ids come from `CUDA_VISIBLE_DEVICES` (default: the first `NUM_GPUS` devices).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_delaywavev4.yaml
```

The best checkpoint (lowest validation denoising loss) is saved as
`<output_dir>/best_val-loss<L>_step<N>/`, and `<output_dir>/best_model` links to that checkpoint. Periodic
checkpoints are saved as `<output_dir>/checkpoint-<step>/`.

**Evaluate.** Arguments are `[CHECKPOINT] [CONFIG] [NUM_VAL_BATCHES]`. An empty checkpoint means
`<output_dir>/best_model` of the config, and `-1` evaluates the full validation split.

```bash
bash eval_ecg_heart.sh "" src/configs/config_ecg_heart_delaywavev4.yaml -1
```

Per-subject predictions are written to `npy_eval/{subid}_{pred,gt}.npy` in the directory that
contains the checkpoint, and geometric metrics (Chamfer and Hausdorff distances, among others) are
appended to `$RESULTS_CSV` (default `eval_metrics.csv` in the same directory). The metrics of the
paper are computed from the `npy_eval` files by `eval_vertex_multiscale.py` (see below). Optional
flags of `src/eval_ecg_mesh.py`, whose defaults are the settings of the paper:

| Flag | Effect |
|---|---|
| `--subject_list <txt>` | evaluate only the listed subjects (one id per line) |
| `--sample_seed <int>` | DDIM sampling seed (default 0) |
| `--eval_batch_size <int>` | override the evaluation batch size of the config |
| `--init_pc_override <npy>` | replace every subject's rest mesh by a template, e.g. a population average (`--init_pc_scale_by_subject` rescales it per subject) |
| `--alpha_delay_override <float>`, `--shuffle_delay` | probe the delay branch: set its gate, or feed it another subject's ECG |

**Visualize.** Arguments are `[CHECKPOINT] [CONFIG] [N_SAMPLES] [OUTPUT_DIR]`.

```bash
bash vis_ecg_heart.sh "" src/configs/config_ecg_heart_delaywavev4.yaml 10
```

**Aggregate metrics across trained variants** (one folder per variant under `--root`):

```bash
python eval_vertex_multiscale.py   --root checkpoints --mesh-info <mesh asset dir> --out outputs
python eval_facearea_multiscale.py --root checkpoints --mesh-info <mesh asset dir> --out outputs
```

`--subject-list <txt>` restricts both scripts to a subgroup of subjects. The whole-heart metrics of the
paper are the `scale == whole` rows of `outputs/vertex_multiscale/metrics-avgstd_per_scale.csv`:

| Paper | Column(s) |
|---|---|
| RV coefficient | `meanRV` |
| RMSE | `predGtL2N_rom` |
| PAR | `peakAmpR_rom` |
| APCR | `xsubjPredRV` / `xsubjGTRV` |
| APDR | `xsDistPred` / `xsDistGT` |

The diversity columns use the `--max-xsubj-subjects` (default 128) validation subjects with the
smallest subject ids.

**Delay tokenization options** (`model_config.delaywavev3_encoder`): `ref_lead` selects the reference
lead (default 1, lead II), and `ref_mode: full` feeds the full lead × lead × scale delay map instead
of a single reference row (`extra_scale` rescales the learning rate of the added columns).

## Reproducing the other results

`reproduce/README.md` maps every result beyond Table 1 to a folder with its code and instructions:
Fig. 1d, the error bars of Table 1, Tables 2 and 3, Fig. 7 (delay tokens in the sEMG decoders
vemg2pose and EMGFormer) and Appendices A.6 and A.7.

## Codebase structure

```
.
├── assets                         # Population-average rest mesh and control-mesh faces for generate_4d_heart.py
├── data_prep                      # UK Biobank exports -> dataset layout, mesh template assets, ECG delay matrix
├── download_ckpts.sh              # Download model weights of Time2Track
├── eval_ecg_heart.sh              # CMD for Time2Track eval given a config yaml
├── eval_facearea_multiscale.py    # Py for surface area metrics
├── eval_vertex_apd.py             # Py for APD per vertex
├── eval_vertex_multiscale.py      # Py for vertex metrics
├── eval_volume_curves.py          # Py for convex volume metrics (unused)
├── example_ecg_gif                # Example of generated 4D heart motion using different methods (Time2Track is denoted by delaywavev4)
├── extract_v4_wavelets.py         # Get wavelet templates in Time2Track
├── generate_4d_heart.py           # 4D heart motion from your own ECG (with or without your rest mesh)
├── libs                           # lib for MDM
├── LICENSE
├── preliminary_ecg.ipynb          # Fig. 1a to 1c (ES timing, inter-lead T-wave delay, PhysCtrl volume curves)
├── plot_heart_surface_scores.py   # Py for plotting metrics on mesh model
├── plot_radar_per_material.py     # Py for plotting radar exp
├── plot_vertex_3d_trajectories.py # Py for plotting 3D trajectory per vertex
├── pyproject.toml
├── README.md
├── reproduce                      # one folder per result beyond Table 1 (see reproduce/README.md)
├── requirements.txt
├── src
│   ├── collect_ecg_delay_maps.py  # Py for extracting learned delay maps
│   ├── configs                    # config yaml files including baselines and different versions of Time2Track (DelayWaveVx)
│   ├── dataset
│   │   └── ukbiobank_dataset.py   # Dataset classes for UKB data
│   ├── eval_ecg_mesh.py           # Py for Time2Track inference
│   ├── eval_ecg_mesh_rest_override.py # Inference variant whose --init_pc_override also takes a per-subject .npz of rest meshes
│   ├── model
│   │   └── ecg_mesh_model.py      # Model classes of Time2Track
│   ├── pipeline_ecg.py            # Pipeline of ECG conditional motion diffusion
│   ├── train_ecg_mesh.py          # Py for Time2Track training
│   ├── utils                      # util codes
│   ├── vis_ecg_mesh.py            # Py for .gif visualization of predictions from Time2Track
├── train_ecg_heart.sh             # CMD for Time2Track training
├── uv.lock
├── vis_delaymap.ipynb             # notebook of learned delay map
└── vis_ecg_heart.sh               # CMD for Time2Track visualizing
```

## Citation

```bibtex
@inproceedings{wei2026time2track,
  title     = {From Heartbeat to Cardiochoreography: A Mechanics Foundation Model for Individualized 4D Cardiac Motion Generation Conditioned on Electrophysiology},
  author    = {Wei, Ziquan and Dan, Tingting and Wu, Guorong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code of this repository is released under the MIT License (`LICENSE`). Third-party code under
`libs/` and the external `TeCh/` and `PhysioWave/` repositories keep their own licenses.

## Acknowledgements

This codebase is based on the official [PhysCtrl](https://github.com/cwchenwang/physctrl) implementation.
