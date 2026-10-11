
from PIL import Image
from typing import Tuple

import trimesh
import numpy as np

def load_mesh(path):
    mesh = trimesh.load_mesh(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh

def paste_image(A: Image.Image, B: Image.Image, h: int, w: int) -> Image.Image:
    A = A.convert("RGBA")
    B = B.convert("RGBA")  # Ensure B has an alpha channel

    A_width, A_height = A.size
    B_width, B_height = B.size

    # Crop A if h or w are negative
    crop_left = max(0, -w)
    crop_top = max(0, -h)
    A_cropped = A.crop((crop_left, crop_top, A_width, A_height))

    # Adjust destination position on B
    paste_x = max(0, w)
    paste_y = max(0, h)

    # Ensure A_cropped fits within B bounds
    max_w = B_width - paste_x
    max_h = B_height - paste_y
    A_cropped = A_cropped.crop((0, 0, min(A_cropped.width, max_w), min(A_cropped.height, max_h)))

    # Use alpha channel of A as mask
    alpha = A_cropped.split()[-1]
    B.paste(A_cropped, (paste_x - 2, paste_y - 2), mask=alpha)

    return B