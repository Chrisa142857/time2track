import sys
sys.path.append('../')
import random
import numpy as np
import torch
import argparse
import os 
import h5py
import genesis as gs
import gc
import json
import trimesh

from utils.seeding import seed_everything
from utils.loading import load_mesh
from utils.visualization import save_pointcloud_video_wdp as save_pointcloud_video
from utils.sample import sample_points_on_mesh, sample_direction_hemisphere
from utils.transform import normalize_points
from torch_cluster import fps

from simulator.mpm.mpm_solver_warp import MPM_Simulator_WARP
from simulator.mpm.decode_param import decode_param_json
from simulator.mpm.filling import get_particle_volume
from tqdm import tqdm

from multiprocessing import Process

# Method 1: Accurate barycentric coordinate calculation
def compute_barycentric_coordinates(point, triangle):
    """
    Compute barycentric coordinates of a point with respect to a triangle.
    
    Args:
        point: 3D point coordinates
        triangle: 3x3 array with triangle vertices
    
    Returns:
        3D barycentric coordinates (u, v, w)
    """
    v0, v1, v2 = triangle
    # Vectors from v0 to other vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0
    
    # Compute dot products
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    
    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        # Handle degenerate triangles
        return np.array([1/3, 1/3, 1/3])
        
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, v, w])

def sample_points_with_face_tracking(mesh, n_points):
    """Sample points on mesh and track their face and barycentric coordinates accurately"""
    # Sample points
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    # Get the triangles for each sampled point
    triangles = mesh.triangles[face_indices]
    
    # Calculate accurate barycentric coordinates for each point
    barycentric_coords = []
    for point, triangle in zip(points, triangles):
        barycentric = compute_barycentric_coordinates(point, triangle)
        barycentric_coords.append(barycentric)
    
    return points, face_indices, np.array(barycentric_coords)

# Function to find corresponding points on transformed mesh
def find_corresponding_points(transformed_mesh, face_indices, barycentric_coords):
    """Find corresponding points on the transformed mesh using face indices and barycentric coordinates"""
    transformed_triangles = transformed_mesh.triangles[face_indices]
    transformed_points = []
    
    for triangle, barycentric in zip(transformed_triangles, barycentric_coords):
        # Apply barycentric coordinates to transformed triangle
        transformed_point = barycentric[0] * triangle[0] + barycentric[1] * triangle[1] + barycentric[2] * triangle[2]
        transformed_points.append(transformed_point)
    
    return np.array(transformed_points)

def run_generation(args):

    device = 'cuda'
    gs.init(backend=gs.cuda)

    # Temporally fixed parameters
    N = 2048
    drag_size_scalar = 0.4
    center = [5, 5, 5]
    drag_size = [drag_size_scalar, drag_size_scalar, drag_size_scalar]
    
    # Load objects config & get obj list 
    if args.dataset_type == 'objaverse':
        data_dir = f'{args.base_dir}/{args.dataset_type}/raw/hf-objaverse-v1/glbs'
        with open(args.uid_list, "r") as f:
            obj_list = json.load(f)
        suffix = '.glb'
    else:
        raise NotImplementedError()
        
    start_idx = max(args.start_idx, 0)
    end_idx = min(args.end_idx, len(obj_list))
    idx_list = list(range(start_idx, end_idx))
    random.shuffle(idx_list)

    output_dir = f'{args.base_dir}/{args.dataset_type}/{args.output_dir}'
    os.makedirs(f'{output_dir}/h5', exist_ok=True)
    import pdb; pdb.set_trace();
    if args.visualization:
        os.makedirs(f'{output_dir}/visualization', exist_ok=True)

    for i in idx_list:
        
        obj_path = obj_list[i]
        if not os.path.exists(f'{data_dir}/{obj_path}{suffix}'):
            continue
        
        jdx_list = list(range(args.start_idx_video, args.end_idx_video))
        random.shuffle(jdx_list)
        
        for j in jdx_list:
            
            torch.cuda.empty_cache()
            gc.collect() 
            
            output_idx = f'{i:05d}_{j:03d}'
            print(f'Generating {output_idx}...')
            output_path = f'{output_dir}/h5/{output_idx}.h5'
            print(output_path)
            if os.path.exists(output_path):
                continue
            
            seed_everything(output_idx)
            
            noise = np.random.randn(3) * 0.01 
            mesh = load_mesh(f'{data_dir}/{obj_path}{suffix}') 
            mesh.vertices, R = normalize_points(mesh.vertices, size=1, output_center=center, random_rotation='simple')
            mesh.vertices = mesh.vertices - 5.0 # + noise
            mesh.export(f'{data_dir}/{obj_path}_normalized{suffix}')
            
            points, face_indices, barycentric_coords = sample_points_with_face_tracking(mesh, N * 20)
            ratio = N / points.shape[0]
            if ratio >= 1 or points.shape != barycentric_coords.shape:
                continue

            points = torch.tensor(points, dtype=torch.float32, device=device).contiguous()
            face_indices = torch.tensor(face_indices, dtype=torch.long, device=device).contiguous()
            barycentric_coords = torch.tensor(barycentric_coords, dtype=torch.float32, device=device).contiguous()

            idx = fps(points, ratio=ratio, random_start=True)
            points = points[idx].cpu().numpy()
            face_indices = face_indices[idx].cpu().numpy()
            barycentric_coords = barycentric_coords[idx].cpu().numpy()

            min_height = np.min(mesh.vertices[:, 1])
            floor_height = np.random.uniform(0.2, min_height + 4.99) 
            mat_type = 3
            gravity = 1 
            E = 0
            nu = 0

            scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                        dt=0.0417, 
                        substeps=20,
                        gravity=(0, -9.8, 0),  
                    ),
                show_viewer=False,
                show_FPS=False
            )
                
            plane = scene.add_entity(gs.morphs.Plane(pos=(0, floor_height, 0), normal=(0, 1, 0)))
            try:
                object = scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file=f'{data_dir}/{obj_path}_normalized{suffix}',
                        scale=1.0,
                        pos=(5, 5, 5), 
                        euler=(0.0, 0.0, 0.0),
                    )
                )
            except:
                continue
  
            scene.build()

            links = object._links
            vert_list = [links[0].get_vverts().cpu().numpy()]
            for i in tqdm(range(48)):
                scene.step()
                scene.visualizer.update_visual_states()
                vert_list.append(links[0].get_vverts().cpu().numpy())

            x_list = []
            drag_point_list = []
            drag_mask_list = []
            drag_force_list = []  

            for i in range(49):
                mesh.vertices = vert_list[i]
                transformed_points = find_corresponding_points(mesh, face_indices, barycentric_coords)
                x_list.append(transformed_points)

            x_list = np.stack(x_list, axis=0)
            v_list = np.zeros((x_list.shape[0], x_list.shape[1], 3), dtype=np.float32)
            F_list = np.zeros((x_list.shape[0], x_list.shape[1], 9), dtype=np.float32)
            C_list = np.zeros((x_list.shape[0], x_list.shape[1], 9), dtype=np.float32)
            vol = np.zeros((x_list.shape[1]), dtype=np.float32)
 
            f = h5py.File(output_path, 'w')
            f.create_dataset('x', data=x_list)
            f.create_dataset('v', data=v_list)
            f.create_dataset('F', data=F_list)
            f.create_dataset('C', data=C_list)
            f.create_dataset('vol', data=vol)
            f.create_dataset('floor_height', data=floor_height)
            f.create_dataset('drag_size', data=drag_size_scalar)
            f.create_dataset('drag_point', data=drag_point_list)
            f.create_dataset('drag_mask', data=drag_mask_list)
            f.create_dataset('drag_force', data=drag_force_list)
            f.create_dataset('E', data=E)
            f.create_dataset('nu', data=nu)
            f.create_dataset('mat_type', data=mat_type)
            f.create_dataset('gravity', data=gravity)
            f.close()
            
            if args.visualization:
                save_pointcloud_video(x_list, [], f'{output_dir}/visualization/{output_idx}.gif', 
                    grid_lim=10, vertical_axis='y')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='outputs_genesis')
    parser.add_argument('--dataset-type', type=str, default='objaverse')
    parser.add_argument('--config', type=str, default='./config/objects.json')
    parser.add_argument('--uid_list', type=str, default='configs/objaverse_valid_uid_list_example.json')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=1)
    parser.add_argument('--start-idx-video', type=int, default=20)
    parser.add_argument('--end-idx-video', type=int, default=21)
    parser.add_argument('--visualization', action='store_true') 
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    
    if args.loop:
        while True:
            p = Process(target=run_generation, args=(args,))
            p.start()
            p.join()
            
            if p.exitcode != 0:
                print("Generation process crashed, restarting...")
            else:
                print("Generation process finished.")
    else:
        run_generation(args)
                    