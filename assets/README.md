# Assets of `generate_4d_heart.py`

| File | Content |
|---|---|
| `population_rest_mesh.npy` | (388, 3) float32, the population-average rest (end-diastolic) heart: the mean of the centred frame-0 control meshes of the 32,369 UK Biobank training subjects, built by `reproduce/cardiac/ecg_only/build_template.py`. It is the rest mesh of the ECG-only mode and of Table 3. |
| `control_mesh_faces.npy` | (390, 4) int64, the quad faces of the 388-point control mesh of biv-volumetric-meshing (https://github.com/cdttk/biv-volumetric-meshing), shared by every subject. |

Both are aggregate or topology data and contain no individual subject.
