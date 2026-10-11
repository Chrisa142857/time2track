import numpy as np
import torch

from PIL import Image

def project_to_image(points_camera, intrinsic_matrix):
    """
    Project 3D points in camera coordinates to 2D image plane.
    :param points_camera: Nx3 array of 3D points in camera coordinates.
    :param intrinsic_matrix: 3x3 camera intrinsic matrix.
    :return: Nx2 array of 2D pixel coordinates.
    """
    # Get homogeneous image coordinates
    points_image_h = intrinsic_matrix @ points_camera.T  # 3xN
    # Normalize to get 2D pixel coordinates
    points_image = points_image_h[:2, :] / points_image_h[2, :]
    return points_image.T  # Nx2

def get_pad(image, target_width=720):
    
    # Get the current size
    if image.ndim == 2:  # Grayscale image
        _, width = image.shape
        channels = None
    elif image.ndim == 3:  # RGB or RGBA image
        _, width, channels = image.shape
    else:
        raise ValueError("Input image must be 2D or 3D (grayscale, RGB, or RGBA).")

    # Desired size
    target_width = 720

    # Calculate padding
    padding_left = (target_width - width) // 2
    padding_right = target_width - width - padding_left

    # Apply padding
    if channels:  # RGB or RGBA image
        padded_image = np.pad(
            image,
            pad_width=((0, 0), (padding_left, padding_right), (0, 0)),
            mode='constant',
            constant_values=2
        )
    else:  # Grayscale image
        padded_image = np.pad(
            image,
            pad_width=((0, 0), (padding_left, padding_right)),
            mode='constant',
            constant_values=2
        )

    return padded_image

def find_and_remove_nearest_point(target, candidates, dense=False):
    
    offset_x = 5.0543
    offset_y = 3.3152
        
    x_min, x_max = target[0] - offset_x, target[0] + offset_x
    y_min, y_max = target[1] - offset_y, target[1] + offset_y
    satisfied_idx = np.where((candidates[:, 0] >= x_min) & (candidates[:, 0] <= x_max) & (candidates[:, 1] >= y_min) & (candidates[:, 1] <= y_max))[0]
    if satisfied_idx.shape[0] == 0:
        return None, candidates
    
    satisfied_candidates = candidates[satisfied_idx]
    distance = np.linalg.norm(satisfied_candidates[:, :2] - target, axis=1)
    min_idx = np.argmin(distance)
    candidate = satisfied_candidates[min_idx]
    kept_idx = np.where(candidates[:, -1] != candidate[-1])
    updated_candidates = candidates[kept_idx]
    return candidate, updated_candidates

def track_first(projected_points, image_shape):
    
    candidate_list = []
    # Fill image with XYZ values
    for i, (x, y, z) in enumerate(projected_points):
        candidate_list.append(np.array([x, y, z, i]))
    candidate_list = np.stack(candidate_list)
    return candidate_list