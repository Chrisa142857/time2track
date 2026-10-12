# Data preparation

These scripts turn the UK Biobank exports into every input of the cardiac experiments.

1. The dataset layout read by `src/dataset/ukbiobank_dataset.py` (`dataset.ecg_dir`, `dataset.mesh_dir`,
   `dataset.npy_dir` in `src/configs/config_ecg_heart*.yaml`). This is the data of all UK Biobank
   results, described in Appendix "Dataset preprocessing": "Resting 12-lead ECG (500Hz) is cropped to a
   single 600-sample window aligned to the cardiac MRI cycle and $z$-normalized per channel. Cardiac
   meshes have $N{=}388$ control points and are sampled to $F{=}50$ frames per cycle."
2. The mesh template assets `control_materials.npy`, `control_material_idmap.json` and `control_face.ply`,
   read by `eval_vertex_multiscale.py` and `eval_facearea_multiscale.py` (`--mesh-info`) and by
   `src/eval_ecg_mesh.py` (`--material_npy`, LVEF and RVEF). They define the heart regions of the radar charts of
   inter-subject diversity ("The metrics are computed across five distinct anatomical regions of the
   heart preprocessed by [Ugurlu et al.]").
3. `all_ecg_delay_matrix.npz`, the per-subject 12 x 12 inter-lead QRS, T-peak and T-offset delay
   matrices. It is `dataset.delay_mat_path` of `config_ecg_heart_delay.yaml` (Delay-T, "the CNN encoder
   augmented with an explicit $12\times12$ inter-lead T-peak delay matrix") and
   `config_ecg_heart_delay_qrs.yaml` (Delay-QRS), and the input of `preliminary_ecg.ipynb` for Fig. 1b
   ("The ECG inter-lead T wave delay of subjects with ES occurred earlier than 512ms is compared with
   subjects entering the ES stage later than 640ms using $t$-test").

## Inputs

UK Biobank data are available to approved researchers at https://www.ukbiobank.ac.uk. The scripts expect
the bulk files in sub-folders one level below a root, the layout of the UK Biobank bulk download used
here (`<root>/<sub>/<file>`).

| input | UK Biobank field | used by |
|---|---|---|
| cardiac MR meshes of Ugurlu et al., *PLOS ONE* 20(7): e0327158, 2025, `<eid>_30099_2_0.zip`, 50 frames per cycle | 30099 | `unzip_meshes.py`, `align_ecg_to_mri.py` (names only) |
| resting 12-lead ECG, `<eid>_20205_<instance>_0.xml` | 20205 | `align_ecg_to_mri.py` |
| heart rate table, CSV with an `eid` index column and columns `12673-<instance>.<array>` (ukbconv naming) | 12673 | `align_ecg_to_mri.py` |

Third-party code (not vendored). The control mesh connectivity, the subdivision matrix and the surface
materials come from biv-volumetric-meshing, https://github.com/cdttk/biv-volumetric-meshing, commit
`aedae41e74434fcee67c92789ca9cedb774706b2`. Only the text files in `src/surface/model/` are read.

```bash
git clone https://github.com/cdttk/biv-volumetric-meshing.git
git -C biv-volumetric-meshing checkout aedae41e74434fcee67c92789ca9cedb774706b2
export BIV_MESHING_DIR=$PWD/biv-volumetric-meshing      # or pass --model-dir <clone>/src/surface/model
pip install neurokit2==0.2.13                           # align_ecg_to_mri.py and ecg_delay_matrix.py
```

The other dependencies (numpy, scipy, pandas, tqdm) are those of the repository.

## Commands

Run from the repository root. The defaults write the layout that the configs expect (their paths are
relative to `src/`, so `../data` is `<repo>/data`).

```bash
# 1. Unzip the mesh archives: <dst>/<sub>/<eid>_30099_2_0/*_model_frame_<k>.txt (4 processes)
python data_prep/unzip_meshes.py --src <mesh zip root> --dst <unzipped mesh root>

# 2. One ASCII PLY per frame: data/ukb_heartmesh_processed/ctrl_mesh_ply/<eid>_30099_2_0/frame_<k>.ply
python data_prep/mesh_to_ply.py --root <unzipped mesh root>

# 3. MRI-aligned resting ECG: data/ukb_heartmesh_processed/ecg_resting_csv/<eid>_<instance>_<k>.csv
python data_prep/align_ecg_to_mri.py --mesh-root <mesh zip root> --ecg-root <resting ECG XML root> \
    --hr-csv <field 12673 CSV>

# 4. frames.npy per subject (ctrl_mesh/) and one z-scored ECG per participant (ecg_resting/)
python data_prep/precompute_npy.py --ecg_dir data/ukb_heartmesh_processed/ecg_resting_csv

# 5. Mesh template assets in ukb_heart_mesh_visual/
python data_prep/control_materials.py
cp "$(ls -d data/ukb_heartmesh_processed/ctrl_mesh_ply/*/ | head -n 1)frame_000.ply" \
    ukb_heart_mesh_visual/control_face.ply

# 6. Inter-lead delay matrices: ukb_heart_mesh_visual/all_ecg_delay_matrix.npz
python data_prep/ecg_delay_matrix.py
```

Outputs:

```
data/ukb_heartmesh_processed/
  ctrl_mesh_ply/<eid>_30099_2_0/frame_000.ply ...   dataset.mesh_dir
  ctrl_mesh/<eid>_30099_2_0/frames.npy              dataset.npy_dir, (50, 388, 3) float32
  ecg_resting_csv/<eid>_2_<k>.csv                   intermediate, (12, 600) microvolts
  ecg_resting/<eid>.npy                             dataset.ecg_dir, (12, 600) float32
ukb_heart_mesh_visual/
  control_materials.npy                             (388,) material id per control point
  control_material_idmap.json                       material name -> id
  control_face.ply                                  388 vertices, 780 edges, 390 quad faces
  all_ecg_delay_matrix.npz                          QRS, T_peak, T_offset (K, 12, 12) ms, filename (K,)
```

Pass `--mesh-info ukb_heart_mesh_visual` to the two metric scripts and
`--material_npy ../ukb_heart_mesh_visual/control_materials.npy` (the default, relative to `src/`) to
`src/eval_ecg_mesh.py`. `preliminary_ecg.ipynb` opens `all_ecg_delay_matrix.npz` in its working
directory, so place or link the file there. The data loader caches the paired subject list and the
displacement filter in hidden JSON files in `dataset.npy_dir` (`.subject_pair_cache.json`,
`.subject_filter_cache_*.json`). Delete them whenever the data change.

## What each step does

**Meshes (steps 1, 2, 4).** Each zip holds 50 `*_model_frame_<k>.txt` files (CSV `x,y,z,Frame`). Step 2
keeps the first three columns and writes the 388 control points with the 780 edges and 390 boundary quad
faces of the hexahedral control mesh (`control_mesh_connectivity.txt`, a face is on the boundary when
only one element has it). A subject with any frame that does not have 388 points is skipped and listed
in `failed_extractions.log` (`--failed-log`). Step 4 stacks the first 50 PLY frames into `frames.npy`.

**ECG alignment (step 3).** The code reads the 600-sample, 500 Hz median beat (`MedianSamples`) of each
XML and scales it by its `Resolution` to microvolts. It does not crop or resample. On every lead,
NeuroKit2 `ecg_clean` and `ecg_peaks` (method `neurokit`) give the first R-peak, the median over leads is
the consensus R-peak (argmax of the lead with the largest range if no lead yields one), and the beat is
rolled circularly so that this R-peak is sample 0 (the stated purpose in the code is the temporal
alignment with the mesh cycle). The heart rate of the
mesh instance (mean of the non-missing `12673-<instance>.*` values) only decides whether the pair is
written, since a mesh without a heart rate is skipped. The stretched time axis of 60 / heart rate seconds
is computed but not saved. One CSV is written per mesh and ECG file of a participant, named
`<eid>_<mesh instance>_<k>`, where `k` counts the participant's resting ECG files in directory-listing
order (instances 2 and 3 of field 20205). Step 4 z-scores every lead
(`(x - mean) / (std + 1e-8)`, float32) and writes `<eid>.npy`. All CSVs of one participant map to the
same file, and the first one written by the worker pool is kept.

**Template assets (step 5).** For each control point, `control_materials.py` collects the surface
vertices with subdivision weight above 1e-6 (`subdivision_matrix.txt`), the surface triangles that touch
them (`ETIndicesSorted.txt`), and assigns the most common triangle material (`ETIndicesMaterials.txt`).
The PLY copied to `control_face.ply` has the edges and faces shared by every subject. The metric scripts
read only its faces, so the subject chosen does not matter.

**Delay matrices (step 6).** Input is `ecg_resting/<eid>.npy`. Per lead, R is the argmax of the 5 to 15 Hz
band-passed, NeuroKit2-cleaned beat in the first 150 ms. T-peak and T-offset come from
`nk.ecg_delineate(method="dwt")` with that R (first detection after R and at least 50 ms before the end),
with band-pass (1 to 7 Hz) fallbacks when the delineation finds none. `delay[i, j]` is
`(peak_j - peak_i) / fs * 1000` ms, wrapped circularly to [-300, 300) samples, so a positive value means
that lead i peaks first. `filename` holds `<eid>.npy` in directory-listing order, and the loader matches
rows by this name. With NeuroKit2 0.2.13 the script reproduces the original matrices exactly (checked on 40 random subjects).

## Scale and runtime

All steps run on CPU. In the original run there were 53,224 mesh archives, 48,308 subjects with PLY
frames, 48,271 with `frames.npy`, and 44,704 participants with an ECG. Pairing ECG and mesh in the
loader gave the $n=40,577$ subjects of the paper, and all of them have a row in the delay file. Step 6
took about 50 min for 44,704 ECGs in one process. Step 5 took 5.5 min.
