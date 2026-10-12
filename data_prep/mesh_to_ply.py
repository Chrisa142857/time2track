"""Export the 388 control points of every unzipped mesh frame as ASCII PLY files.

Input: the folders written by unzip_meshes.py. Each leaf folder holds one
`*_model_frame_<k>.txt` file per frame (CSV with header x,y,z,Frame).

Output: <output-dir>/<subject folder>/frame_<k>.ply with 388 vertices, the
edges and the boundary quad faces of the hexahedral control mesh of
biv-volumetric-meshing (control_mesh_connectivity.txt). This is the layout of
`dataset.mesh_dir` in src/configs/config_ecg_heart*.yaml. Subjects whose frames
do not all have 388 control points are skipped and listed in --failed-log.
"""
import argparse
import os
import re
import warnings
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", message="loadtxt: input contained no data")

REPO = Path(__file__).resolve().parents[1]
EXPECTED_CONTROL_POINTS = 388

# Local node indices for the 6 quad faces of a hexahedral element.
# Node ordering follows xi_order: (xi1,xi2,xi3) corners
# [0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]
HEX_FACE_LOCAL_INDICES = [
    (0, 1, 3, 2),  # xi3=0 face
    (4, 5, 7, 6),  # xi3=1 face
    (0, 1, 5, 4),  # xi2=0 face
    (2, 3, 7, 6),  # xi2=1 face
    (0, 2, 6, 4),  # xi1=0 face
    (1, 3, 7, 5),  # xi1=1 face
]


def frame_index(name: str) -> int:
    match = re.search(r"_frame_(\d+)", name)
    return int(match.group(1)) if match else -1


def list_subject_dirs(root: Path) -> list[Path]:
    """Walk root and return all leaf directories that contain frame txt files."""
    subject_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any("_frame_" in f and f.endswith(".txt") for f in filenames):
            subject_dirs.append(Path(dirpath))
            dirnames.clear()  # don't recurse deeper once we found frame files
    return subject_dirs


def load_frames_from_dir(subject_dir: Path, frame_limit: int) -> list[np.ndarray]:
    """Load up to frame_limit frames from a subject directory of frame txt files."""
    frame_files = sorted(
        [f for f in subject_dir.iterdir() if "_model_frame_" in f.name and f.suffix == ".txt"],
        key=lambda p: frame_index(p.name),
    )
    selected = frame_files[:frame_limit]

    frames = []
    for frame_path in selected:
        frames.append(np.loadtxt(frame_path, delimiter=",", skiprows=1))
    return frames


def validate_frame_control_points(frames: list[np.ndarray]) -> None:
    for frame_idx, frame in enumerate(frames):
        if frame.shape[0] != EXPECTED_CONTROL_POINTS:
            raise ValueError(
                f"Frame {frame_idx:03d} has {frame.shape[0]} control points, "
                f"expected {EXPECTED_CONTROL_POINTS}."
            )
        if frame.shape[1] < 3:
            raise ValueError(
                f"Frame {frame_idx:03d} has shape {frame.shape}; expected at least 3 columns."
            )


def load_control_connectivity(path: Path) -> np.ndarray:
    """Load (K, 8) hex element connectivity and convert to 0-based indices."""
    return np.loadtxt(path, dtype=int) - 1


def extract_hex_boundary_faces(connectivity: np.ndarray) -> np.ndarray:
    """Return boundary (exterior) quad faces from hex elements as a (F, 4) array.

    A face is exterior if it belongs to exactly one hex element.
    """
    face_map: dict[tuple, list[tuple]] = defaultdict(list)
    for elem_nodes in connectivity:
        for local_face in HEX_FACE_LOCAL_INDICES:
            oriented = tuple(int(elem_nodes[i]) for i in local_face)
            canonical = tuple(sorted(oriented))
            face_map[canonical].append(oriented)

    boundary = [faces[0] for faces in face_map.values() if len(faces) == 1]
    return np.array(boundary, dtype=int)


def extract_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from a face array (N, M) of any polygon size."""
    edges: set[tuple[int, int]] = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edges.add((min(a, b), max(a, b)))
    return np.array(sorted(edges), dtype=int)


def export_frames_to_ply(
    frames: list[np.ndarray],
    faces: np.ndarray,
    edges: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """Write one PLY per frame: 388 control point vertices, boundary quad faces, edges."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_faces = len(faces)
    n_edges = len(edges)

    ply_paths = []
    failed = []
    for frame_idx, frame in enumerate(frames):
        ctrl_pts = frame[:, :3]
        if ctrl_pts.shape[0] != EXPECTED_CONTROL_POINTS:
            failed.append((frame_idx, ctrl_pts.shape))
            continue

        ply_path = output_dir / f"frame_{frame_idx:03d}.ply"
        with ply_path.open("w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {EXPECTED_CONTROL_POINTS}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element edge {n_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write(f"element face {n_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for pt in ctrl_pts:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            for e in edges:
                f.write(f"{e[0]} {e[1]}\n")
            for face in faces:
                f.write(f"4 {face[0]} {face[1]} {face[2]} {face[3]}\n")

        ply_paths.append(ply_path)

    if failed:
        raise ValueError(
            f"Frames {[i for i, s in failed]} produced shapes {[s for i, s in failed]}; "
            f"expected ({EXPECTED_CONTROL_POINTS}, 3)."
        )
    return ply_paths


def process_single_subject(
    subject_dir: Path,
    frame_count: int,
    output_dir: Path,
    faces: np.ndarray,
    edges: np.ndarray,
) -> tuple[bool, Path, object]:
    """Process one subject directory: export PLY frames. Returns (success, subject_dir, result_or_error)."""
    target_dir = output_dir / subject_dir.name
    if target_dir.exists() and len(list(target_dir.iterdir())) >= 50:
        return (True, subject_dir, target_dir)

    try:
        frames = load_frames_from_dir(subject_dir, frame_limit=frame_count)
        validate_frame_control_points(frames)
        export_frames_to_ply(frames, faces, edges, target_dir)
        return (True, subject_dir, target_dir)
    except Exception as e:
        return (False, subject_dir, str(e))


def build_parser() -> argparse.ArgumentParser:
    biv = os.environ.get("BIV_MESHING_DIR")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="Root directory of the unzipped mesh folders (output of unzip_meshes.py).")
    parser.add_argument("--frame-count", type=int, default=999,
                        help="Maximum number of frames to export per subject.")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "data" / "ukb_heartmesh_processed" / "ctrl_mesh_ply",
                        help="Folder where the per-subject PLY folders are written.")
    parser.add_argument("--model-dir", type=Path, required=biv is None,
                        default=Path(biv, "src", "surface", "model") if biv else None,
                        help="biv-volumetric-meshing src/surface/model directory "
                             "(default: $BIV_MESHING_DIR/src/surface/model).")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes.")
    parser.add_argument("--failed-log", type=Path, default=Path("failed_extractions.log"),
                        help="Tab-separated list of subjects that failed, with the error.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    subject_dirs = list_subject_dirs(args.root)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject frame directories found under {args.root}")
    print(f"Found {len(subject_dirs)} subjects under {args.root}")

    ctrl_connectivity_path = args.model_dir / "control_mesh_connectivity.txt"
    if not ctrl_connectivity_path.exists():
        raise FileNotFoundError(
            f"Control mesh connectivity file not found: {ctrl_connectivity_path}"
        )

    print(f"Loading control mesh connectivity from {ctrl_connectivity_path}")
    connectivity = load_control_connectivity(ctrl_connectivity_path)
    faces = extract_hex_boundary_faces(connectivity)
    edges = extract_edges_from_faces(faces)
    print(
        f"Control mesh: {EXPECTED_CONTROL_POINTS} nodes, "
        f"{len(connectivity)} hex elements, "
        f"{len(faces)} boundary faces, "
        f"{len(edges)} edges"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed_files: list[tuple] = []
    worker = partial(
        process_single_subject,
        frame_count=args.frame_count,
        output_dir=args.output_dir,
        faces=faces,
        edges=edges,
    )
    with Pool(processes=args.num_workers) as pool:
        for success, subject_dir, result in tqdm(
            pool.imap(worker, subject_dirs),
            total=len(subject_dirs),
            desc="Extracting PLY",
        ):
            if not success:
                failed_files.append((subject_dir, result))

    if failed_files:
        print(f"\nWARNING: {len(failed_files)} subjects failed to process.")
        print("First 5 failures:")
        for s_dir, err in failed_files[:5]:
            print(f"  - {s_dir.name}: {err}")
        with open(args.failed_log, "w") as f:
            for s_dir, err in failed_files:
                f.write(f"{s_dir.name}\t{err}\n")


if __name__ == "__main__":
    main()
