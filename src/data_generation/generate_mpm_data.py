import sys
sys.path.append('../')
 
import numpy as np
import torch
import argparse
import os 
import h5py
import taichi as ti
import warp as wp
import gc
import json

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

def run_generation(args):
    
    device = "cuda"
    wp.config.kernel_cache_dir = f'{args.base_dir}/{args.warp_cache_dir}'
    os.makedirs(wp.config.kernel_cache_dir, exist_ok=True)
    
    wp.init()
    ti.init(arch=ti.cuda, device_memory_GB=8.0)
     
    N = 2048
    center = [5, 5, 5]
    drag_size = [0.4, 0.4, 0.4]
    material_type_list = ['elastic', 'plasticine', 'sand']
    
    if args.material not in material_type_list:
        raise ValueError(f"Invalid material type: {args.material}")
    material_type_index = material_type_list.index(args.material)
    
    # Load objects config & get obj list
    print("Loading objects config...")
    material_params, bc_params, time_params, preprocessing_params, camera_params = decode_param_json(args.config)
    
    if args.dataset_type == 'objaverse':
        data_dir = f'{args.base_dir}/{args.dataset_type}/raw/hf-objaverse-v1/glbs'
        with open(args.uid_list, "r") as f:
            obj_list = json.load(f)
        suffix = '.glb'
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
        
    start_idx = max(args.start_idx, 0)
    end_idx = min(args.end_idx, len(obj_list))
    idx_list = list(range(start_idx, end_idx)) 
 
    output_dir = f'{args.base_dir}/{args.dataset_type}/{args.output_dir}'
    os.makedirs(f'{output_dir}/h5', exist_ok=True)
    if args.visualization:
        os.makedirs(f'{output_dir}/visualization', exist_ok=True)
    
    for i in idx_list:
        
        obj_path = obj_list[i]  
        if not os.path.exists(f'{data_dir}/{obj_path}{suffix}'):
            continue
            
        torch.cuda.empty_cache()
        gc.collect()
        wp.clear_kernel_cache()
        
        output_idx = f'{i:05d}_{material_type_index:03d}'
        print(f'Generating {output_idx}...')
        output_path = f'{output_dir}/h5/{output_idx}.h5'
        if os.path.exists(output_path):
            continue
        
        seed_everything(output_idx)
        
        try:
            mesh = load_mesh(f'{data_dir}/{obj_path}{suffix}')
        except:
            print(f"Failed to load mesh {obj_path}{suffix}")
        
        noise = np.random.randn(3) * 0.01 # Make the model generalizable
        points, face_normals = sample_points_on_mesh(mesh, N * 20)
        ratio = N / points.shape[0]
        if ratio >= 1 or points.shape != face_normals.shape:
            continue
        
        points = points + noise
        points, R = normalize_points(points, size=1, output_center=center, random_rotation='simple') # Randomly rotate around Z-axis
        face_normals = face_normals @ R
        points = torch.tensor(points, dtype=torch.float32, device=device).contiguous()
        face_normals = torch.tensor(face_normals, dtype=torch.float32, device=device).contiguous()
        
        idx = fps(points, ratio=ratio, random_start=True)
        points = points[idx]
        face_normals = face_normals[idx]
        points = points.cpu().numpy()
        face_normals = face_normals.cpu().numpy()
        
        if points.shape[0] != N:
            continue
        
        log_E = np.random.uniform(4, 7)
        E = np.power(10, log_E)
        nu = np.random.uniform(0.05, 0.45) 
        
        material_params["material"] = args.material 
        if args.material == 'elastic':
            force_num = np.random.randint(1, 2) # In our dataset, we only support one non-gravity force for elastic material, feel free to add more
        elif args.material == 'plasticine':
            force_num = np.random.randint(0, 1)
            material_params["yield_stress"] = 10000
        elif args.material == 'sand':
            force_num = np.random.randint(0, 1) 
            material_params["friction"] = 0.3
            
        if args.material == 'elastic':
            floor_height = 0.2 
        else:
            per_grid_height = material_params["grid_lim"] / material_params["n_grid"]
            floor_height = int(points[:, 1].min().item() / per_grid_height) * per_grid_height - 1e-3
            floor_height = np.random.uniform(0.2, floor_height)

        gravity = 0 if force_num > 0 else 1 # Apply gravity if there is no non-gravity force 
        material_params["g"] = [0, -9.8, 0] if gravity == 1 else [0, 0, 0]
                    
        substep_dt = time_params["substep_dt"]
        frame_dt = time_params["frame_dt"]
        frame_num = time_params["frame_num"]
        step_per_frame = int(frame_dt / substep_dt)
        
        material_params["E"] = E
        material_params["nu"] = nu
        grid_lim = material_params["grid_lim"]
        points = torch.tensor(points, dtype=torch.float32, device=device).contiguous()
        particle_volume = get_particle_volume(points, material_params["n_grid"], material_params["grid_lim"] / material_params["n_grid"],
            unifrom=material_params["material"] == "sand").to(device=device)
        
        mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized
        mpm_solver.load_initial_data_from_torch(points, particle_volume, n_grid=material_params["n_grid"],
            grid_lim=material_params["grid_lim"])
        mpm_solver.set_parameters_dict(material_params)
        mpm_solver.finalize_mu_lam_bulk()
        
        # Boundary
        mpm_solver.add_surface_collider([0.2, 0, 0], [1, 0, 0], surface='cut')
        mpm_solver.add_surface_collider([grid_lim-0.2, 0, 0], [-1, 0, 0], surface='cut')
        mpm_solver.add_surface_collider([0, 0, 0.2], [0, 0, 1], surface='cut')
        mpm_solver.add_surface_collider([0, 0, grid_lim-0.2], [0, 0, -1], surface='cut')
        
        # Floor
        mpm_solver.add_surface_collider([0, floor_height, 0], [0, 1, 0], surface='cut')
        mpm_solver.add_surface_collider([0, grid_lim-0.2, 0], [0, -1, 0], surface='cut')
        
        drag_point_list = []
        drag_mask_list = []
        drag_force_list = []
        total_mask = np.zeros_like(particle_volume.cpu().numpy())

        for j in range(force_num):

            force_coeff = np.random.uniform(0.02, 0.2)
            drag_point_idx = np.random.randint(0, N)
            drag_point = points[drag_point_idx].cpu().numpy()
            drag_normal = face_normals[drag_point_idx]
            drag_normal /= np.linalg.norm(drag_normal)
            drag_dir = sample_direction_hemisphere(drag_normal)
            drag_force = drag_dir * force_coeff
            mpm_solver.add_impulse_on_particles(force=np.zeros_like(drag_force), dt=time_params["substep_dt"], 
                point=drag_point, size=drag_size, num_dt=step_per_frame * frame_num)

            drag_mask = wp.to_torch(mpm_solver.impulse_params[-1].mask).cpu().numpy()
            total_mask = np.logical_or(total_mask, drag_mask)
            total_volume = torch.sum(particle_volume)
            masked_volume = torch.sum(particle_volume[drag_mask > 0])
            mean_masked_volume = torch.mean(particle_volume[drag_mask > 0])
            mask_ratio = (masked_volume / total_volume).item()
            base_drag_coeff = 9.8 * material_params["density"] * mean_masked_volume.item() / mask_ratio
            drag_force = drag_force * base_drag_coeff
            
            mpm_solver.add_impulse_on_particles(force=drag_force, dt=time_params["substep_dt"], 
                point=drag_point, size=drag_size, num_dt=step_per_frame * frame_num)
            
            drag_point_list.append(drag_point)
            drag_mask_list.append(drag_mask)
            drag_force_list.append(drag_force) # Save the drag force for visualization
        
        x_list = []
        v_list = []
        F_list = []
        C_list = []
        
        # Add initial x and v
        x = mpm_solver.export_particle_x_to_torch()
        v = mpm_solver.export_particle_v_to_torch()
        F = mpm_solver.export_particle_F_to_torch()
        C = mpm_solver.export_particle_C_to_torch()

        x_list.append(x.cpu().numpy())
        v_list.append(v.cpu().numpy())
        F_list.append(F.cpu().numpy())
        C_list.append(C.cpu().numpy())
        
        for frame in tqdm(range(frame_num)):
            for step in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device="cuda")
            x = mpm_solver.export_particle_x_to_torch()
            v = mpm_solver.export_particle_v_to_torch()
            F = mpm_solver.export_particle_F_to_torch()
            C = mpm_solver.export_particle_C_to_torch()
            x_list.append(x.cpu().numpy())
            v_list.append(v.cpu().numpy())
            F_list.append(F.cpu().numpy())
            C_list.append(C.cpu().numpy())
            
        x_list = np.stack(x_list, axis=0)
        v_list = np.stack(v_list, axis=0)
        F_list = np.stack(F_list, axis=0)
        C_list = np.stack(C_list, axis=0)
        x_inactivate_list = x_list[:, total_mask <= 0]
        x_drag_list = x_list[:, total_mask > 0]
        if x_drag_list.shape[1] == 0:
            x_drag_list = []
        
        if len(drag_point_list) > 0:
            drag_point_list = np.stack(drag_point_list, axis=0)
            drag_mask_list = np.stack(drag_mask_list, axis=0)
            drag_force_list = np.stack(drag_force_list, axis=0)

        f = h5py.File(output_path, 'w')
        f.create_dataset('x', data=x_list) 
        f.create_dataset('v', data=v_list)
        f.create_dataset('F', data=F_list)
        f.create_dataset('C', data=C_list)
        f.create_dataset('vol', data=particle_volume.cpu().numpy())
        f.create_dataset('floor_height', data=floor_height) 
        f.create_dataset('drag_point', data=drag_point_list)
        f.create_dataset('drag_mask', data=drag_mask_list)
        f.create_dataset('drag_force', data=drag_force_list)
        f.create_dataset('E', data=E)
        f.create_dataset('nu', data=nu)
        f.create_dataset('mat_type', data=material_type_index)
        f.create_dataset('gravity', data=gravity)
        f.close()
        
        if args.visualization: 
            save_pointcloud_video(x_inactivate_list, x_drag_list, f'{output_dir}/visualization/{output_idx}.gif', grid_lim=10, vertical_axis='y')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='data') 
    parser.add_argument('--output_dir', type=str, default='outputs_mpm')
    parser.add_argument('--dataset_type', type=str, default='objaverse')
    parser.add_argument('--config', type=str, default='configs/objaverse_mpm.json')  
    parser.add_argument('--uid_list', type=str, default='configs/objaverse_valid_uid_list_example.json')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1) 
    parser.add_argument('--warp_cache_dir', type=str, default='warp_cache') 
    parser.add_argument('--material', type=str, default='elastic')
    parser.add_argument('--visualization', action='store_true')
    args = parser.parse_args()
    
    run_generation(args)
                    