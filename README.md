# Official code for Time2Track

**From Heartbeat to Cardiochoreography: A Mechanics Foundation Model for Individualized 4D Cardiac Motion Generation Conditioned on Electrophysiology**

Ziquan Wei, Tingting Dan, Guorong Wu. NeurIPS 2026.

Project page: https://chrisa142857.github.io/time2track/

### Setup

```
uv sync
```

Two encoders are imported from directories at the repository root that are not part of this
repository: `TeCh/` (TeCh ECG encoder) and `PhysioWave/` (wavelet backbone). Link or copy them there
before using the `tech`, `physiowave` or `delaywave*` encoders. The delaywave configs read the
pretrained PhysioWave ECG weights from `pretrained_path` (default `../physctrl/PhysioWave/ecg.pth`,
relative to `src/`). Dataset locations are config fields as well (`dataset.ecg_dir`,
`dataset.npy_dir`, `dataset.mesh_dir`, relative to `src/`).

## Usage

**Train.** `NUM_GPUS` selects `src/configs/acc/<NUM_GPUS>gpu.yaml` (1, 2, 4, 7 or 8 GPUs), and the GPU
ids come from `CUDA_VISIBLE_DEVICES` (default: the first `NUM_GPUS` devices).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_delaywavev4.yaml
```

The best checkpoint (lowest validation denoising loss) is saved as
`<output_dir>/best_val-loss<L>_step<N>/`, and `<output_dir>/best_model` links to it. Periodic
checkpoints are saved as `<output_dir>/checkpoint-<step>/`.

**Evaluate.** Arguments are `[CHECKPOINT] [CONFIG] [NUM_VAL_BATCHES]`. An empty checkpoint means
`<output_dir>/best_model` of the config, and `-1` evaluates the full validation split.

```bash
bash eval_ecg_heart.sh "" src/configs/config_ecg_heart_delaywavev4.yaml -1
```

Per-subject predictions are written to `npy_eval/{subid}_{pred,gt}.npy` in the directory that
contains the checkpoint, and the metrics are appended to `$RESULTS_CSV` (default `eval_metrics.csv`
in the same directory). Optional flags of `src/eval_ecg_mesh.py`, whose defaults reproduce the paper:

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
python eval_vertex_multiscale.py   --root <variants dir> --mesh-info <mesh asset dir> --out outputs
python eval_facearea_multiscale.py --root <variants dir> --mesh-info <mesh asset dir> --out outputs
```

`--subject-list <txt>` restricts both scripts to a subgroup of subjects.

**Delay tokenization options** (`model_config.delaywavev3_encoder`): `ref_lead` selects the reference
lead (default 1, lead II), and `ref_mode: full` feeds the full lead × lead × scale delay map instead
of a single reference row (`extra_scale` rescales the learning rate of the added columns).

## Repository tooling

- `paths.py` is the single source of file locations for local experiment scripts. Machine-specific
  roots are read from an untracked `.env` (template: `.env.example`).
- `tools/runmeta.py` creates a local run directory, `experiments/YYYY.MM.DD-runNN_slug/`, and records
  its code version, config, input hashes and environment.
- `tools/check_naming.py` (rules in `tools/naming.ini`) checks artifact naming and run layout. Enable
  it, together with the Git LFS hooks, with `git config core.hooksPath .githooks`.

## Codebase structure

```
.
├── .env.example                   # Template for machine-specific roots read by paths.py
├── .githooks                      # pre-commit naming check and Git LFS hooks
├── download_ckpts.sh              # Download model weights of Time2Track
├── eval_ecg_heart.sh              # CMD for Time2Track eval given a config yaml
├── eval_facearea_multiscale.py    # Py for surface area metrics
├── eval_vertex_apd.py             # Py for APD per vertex
├── eval_vertex_multiscale.py      # Py for vertex metrics
├── eval_volume_curves.py          # Py for convex volume metrics (unused)
├── example_ecg_gif                # Example of generated 4D heart motion using different methods (Time2Track is denoted by delaywavev4)
├── extract_v4_wavelets.py         # Get wavelet templates in Time2Track
├── libs                           # lib for MDM
├── paths.py                       # Single source of file locations
├── plot_heart_surface_scores.py   # Py for plotting metrics on mesh model
├── plot_radar_per_material.py     # Py for plotting radar exp
├── plot_vertex_3d_trajectories.py # Py for plotting 3D trajectory per vertex
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── collect_ecg_delay_maps.py  # Py for extracting learned delay maps
│   ├── configs                    # config yaml files including baselines and different versions of Time2Track (DelayWaveVx)
│   ├── dataset
│   │   └── ukbiobank_dataset.py   # Dataset classes for UKB data
│   ├── eval_ecg_mesh.py           # Py for Time2Track inference
│   ├── eval_ecg_mesh_e5.py        # Inference variant whose --init_pc_override also takes a per-subject .npz of rest meshes
│   ├── model
│   │   └── ecg_mesh_model.py      # Model classes of Time2Track
│   ├── pipeline_ecg.py            # Pipeline of ECG conditional motion diffusion
│   ├── train_ecg_mesh.py          # Py for Time2Track training
│   ├── utils                      # util codes
│   ├── vis_ecg_mesh.py            # Py for .gif visualization of predictions from Time2Track
├── tools                          # Run metadata (runmeta.py) and naming checks (check_naming.py, naming.ini)
├── train_ecg_heart.sh             # CMD for Time2Track training
├── uv.lock
├── vis_delaymap.ipynb             # notebook of learned delay map
└── vis_ecg_heart.sh               # CMD for Time2Track visualizing
```

## Acknowledgements

This codebase is based on the official [PhysCtrl](https://github.com/cwchenwang/physctrl) implementation.
