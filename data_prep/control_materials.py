"""Label each of the 388 control points with the material of the surface it drives.

For control point i, the surface vertices with subdivision weight > 1e-6
(subdivision_matrix.txt, 5810 x 388) are collected, every surface triangle
(ETIndicesSorted.txt) touching one of them is gathered, and the most common
material of those triangles (ETIndicesMaterials.txt) becomes the label
(MATERIAL_IDS below, 0 if none).

Writes <out-dir>/control_materials.npy (388,) int and
<out-dir>/control_material_idmap.json (material name -> id), the files read by
eval_vertex_multiscale.py / eval_facearea_multiscale.py (--mesh-info <out-dir>)
and src/eval_ecg_mesh.py (--material_npy).
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import trange

MATERIAL_IDS = {
    "LV_ENDOCARDIAL": 1, "LV_EPICARDIAL": 2, "RV_EPICARDIAL": 3, "RV_FREEWALL": 4,
    "RV_SEPTUM": 5, "MITRAL_VALVE": 6, "MITRAL_VALVE_CUT": 7, "TRICUSPID_VALVE": 8,
    "TRICUSPID_VALVE_CUT": 9, "AORTA_VALVE": 10, "AORTA_VALVE_CUT": 11,
    "PULMONARY_VALVE": 12, "PULMONARY_VALVE_CUT": 13, "THRU_WALL": 14, "EPI_TO_SEPTUM": 15
}


def load_face_materials(materials_path: Path) -> list[str]:
    materials = []
    for line in materials_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        _, label = stripped.split()
        materials.append(label)
    return materials


def compute_control_materials(model_dir: Path) -> np.ndarray:
    subdivision_matrix = np.loadtxt(model_dir / "subdivision_matrix.txt")  # (5810, 388)
    faces = np.loadtxt(model_dir / "ETIndicesSorted.txt", dtype=int) - 1  # (11760, 3)
    face_materials = load_face_materials(model_dir / "ETIndicesMaterials.txt")  # list of 11760 strings

    control_materials = np.zeros(388, dtype=int)

    for i in trange(388):
        # Find influenced surface vertices (non-zero weights)
        influenced_verts = np.where(subdivision_matrix[:, i] > 1e-6)[0]

        # Find faces that include any influenced vertex
        relevant_faces = set()
        for face in faces:
            if any(v in influenced_verts for v in face):
                relevant_faces.add(tuple(face))

        # Collect materials from relevant faces
        materials = [face_materials[faces.tolist().index(list(f))] for f in relevant_faces]

        # Most common material
        if materials:
            most_common = Counter(materials).most_common(1)[0][0]
            control_materials[i] = MATERIAL_IDS.get(most_common, 0)  # 0 for unknown
    return control_materials


def main():
    biv = os.environ.get("BIV_MESHING_DIR")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=biv is None,
                    default=Path(biv, "src", "surface", "model") if biv else None,
                    help="biv-volumetric-meshing src/surface/model directory "
                         "(default: $BIV_MESHING_DIR/src/surface/model).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "ukb_heart_mesh_visual",
                    help="Mesh asset directory (the --mesh-info of the evaluation scripts).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "control_materials.npy", compute_control_materials(args.model_dir))
    (args.out_dir / "control_material_idmap.json").write_text(json.dumps(MATERIAL_IDS, indent=4))
    print(f"Saved control_materials.npy and control_material_idmap.json to {args.out_dir}")


if __name__ == "__main__":
    main()
