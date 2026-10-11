import torch
import numpy as np

def generate_rotation_matrix():
    
    # Generate random angles for rotation
    alpha, beta, gamma = 2 * np.pi * np.random.rand(3)

    # Rotation matrix around the X-axis (alpha)
    Rx = np.array([
        [1,              0,               0],
        [0,  np.cos(alpha), -np.sin(alpha)],
        [0,  np.sin(alpha),  np.cos(alpha)]
    ])

    # Rotation matrix around the Y-axis (beta)
    Ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [              0, 1,              0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation matrix around the Z-axis (gamma)
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma),  np.cos(gamma), 0],
        [              0,                0,   1]
    ])

    # Full rotation: Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    return R

def generate_rotation_matrix_simple():
    
    beta = 2 * np.pi * np.random.rand(1).item()
    Ry = np.array([
        [ np.cos(beta), 0, np.sin(beta)],
        [              0, 1,              0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    return Ry

def normalize_points(points, size=0.5, output_center=[0, 0, 0], random_rotation='simple'):
    if random_rotation == 'full':
        R = generate_rotation_matrix()
        points = points @ R.T
    elif random_rotation == 'simple':
        R = generate_rotation_matrix_simple()
        points = points @ R.T
    else:
        R = np.eye(3)
    bmax = points.max(axis=0)
    bmin = points.min(axis=0)
    aabb = bmax - bmin
    center = (bmax + bmin) / 2
    points = size * (points - center) / (aabb.max() * 0.5)
    points += np.array(output_center)
    return points, R

def transform2origin(v, size=1):
    bmax = v.max(axis=0)
    bmin = v.min(axis=0)
    aabb = bmax - bmin
    center = (bmax + bmin) / 2
    scale = size / (aabb.max() * 0.5)
    new_v = (v - center) * scale
    return new_v, center, scale 

def shift2center_th(position_tensor, center=[5, 5, 5]):
    tensor = torch.tensor(center, dtype=torch.float32, device=position_tensor.device).contiguous()
    return position_tensor + tensor

def shift2center(position_tensor, center=[5, 5, 5]):
    tensor = np.array(center)
    return position_tensor + tensor

