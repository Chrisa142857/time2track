import os
import argparse
import json
import sys
import gc
import random
import warp as wp

sys.path.append("../libs")
sys.path.append("../libs/LGM")
sys.path.append("../libs/vggt")
sys.path.append("../libs/das")

import numpy as np
import trimesh
import torch    
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2

import h5py
import tyro
import imageio
import open3d as o3d

from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA 
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection 
from diffusers import AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler
from diffusers.utils import export_to_gif, export_to_video
from kiui.cam import orbit_camera
from safetensors.torch import load_file
from torch_cluster import fps
from omegaconf import OmegaConf 

from sv3d.diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline
from LGM.core.models import LGM
from LGM.core.options import AllConfigs 
from LGM.core.gs import GaussianRenderer
from LGM.mvdream.pipeline_mvdream import MVDreamPipeline

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from utils.seeding import seed_everything
from utils.track_utils.preprocessing import track_first, find_and_remove_nearest_point
from utils.track_utils.visualize_tracks import visualize_tracks 
from utils.interpolate import *
from utils.loading import paste_image
from utils.image_process import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image
from utils.transform import transform2origin, shift2center_th
from utils.sim_utils import get_particle_volume 

# Diffusion
from model.spacetime import MDM_ST
from pipeline_traj import TrajPipeline
from options import TestingConfig

device = torch.device("cuda")

def run_track(args, output_dir):
    
    N = 2048
    frame_num = 49

    animated_points = np.load(f'{output_dir}/gen_data.npy')
    animated_points = animated_points * 2 
    new_animate_points = np.zeros((frame_num, N, 3))
    for i in range(frame_num - 2): # Interpolate since we only generate 24 frames
        if i % 2 == 0:  
            new_animate_points[i + 1] = animated_points[i // 2]
        else:
            new_animate_points[i + 1] = (animated_points[i // 2] + animated_points[i // 2 + 1]) / 2
    new_animate_points[0] = new_animate_points[1]
    new_animate_points[frame_num - 1] = new_animate_points[frame_num - 2]
    animated_points = new_animate_points
 
    projection_matrix = np.load('templates/projection.npy')
    crop_info = np.load(f'{output_dir}/crop_info.npy')
    center = np.load(f'{output_dir}/center.npy')
    scale = np.load(f'{output_dir}/scale.npy') 
    animated_points = (animated_points / scale) + center    

    # Aligned to Gaussian points at this moment
    sys.argv = ['pipeline_track_gen.py', 'big']
    opt = tyro.cli(AllConfigs)

    scale_factor = 1
    focal = 0.5 * opt.output_size / np.tan(np.deg2rad(opt.fovy) / 2)
    new_fovy_rad = scale_factor * np.arctan(opt.output_size / focal)
    new_fovy_deg = np.rad2deg(new_fovy_rad)
    opt.fovy = new_fovy_deg
    opt.output_size *= scale_factor # Expand canvas size by 2

    gs = GaussianRenderer(opt)
    gaussians = gs.load_ply(f'{output_dir}/point_cloud.ply', compatible=True).to(device).float()
    idx = torch.from_numpy(np.load(f'{output_dir}/fps_idx.npy')).to(device)
    gaussian_pos = gaussians[:, :3].contiguous()
    drive_x = gaussian_pos[idx]
    cdist = -1.0 * torch.cdist(gaussian_pos, drive_x) # [N, 2048]
    _, topk_index = torch.topk(cdist, 8, -1)

    cam_poses = torch.from_numpy(orbit_camera(0, 0, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
    cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
    cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
    cam_view_proj = cam_view @ gs.proj_matrix.to(device) # [V, 4, 4]
    cam_pos = - cam_poses[:, :3, 3] # [V, 3]

    pos = []
    frames = []
    input_raw = np.array(Image.open(f'{args.base_dir}/{args.data_name}/input.png'))
    input_mask = np.array(Image.open(f'{output_dir}/input_mask.png').convert('L'))
    input_raw[input_mask != 0] = 0  # Set masked pixels (where mask is 0) to black
    input_raw = Image.fromarray(input_raw)
  
    for i in tqdm(range(0, frame_num, 1)):
        drive_current = torch.from_numpy(animated_points[i]).to(device).float()
        ret_points, new_rotation = interpolate_points(gaussian_pos, gaussians[:, 7:11], drive_x, drive_current, topk_index)
        gaussians_new = gaussians.clone()
        gaussians_new[:, :3] = ret_points
        gaussians_new[:, 7:11] = new_rotation
        pos.append(ret_points.cpu().numpy()) 
    
    track_template = np.load(f'templates/tracks_template.npy', allow_pickle=True) 
    tracks = track_template.item()['tracks']
    tracks_output = tracks.copy()
    tracks_init = tracks[0, 0]  
    track_idx = []
    mask = np.zeros(tracks_init.shape[0], dtype=bool)

    h_begin, w_begin, res = crop_info[0], crop_info[1], crop_info[2]
    image_shape = (res, res)  # Example image shape (H, W)

    drag_points = []

    for i in tqdm(range(frame_num)):
        
        points = pos[i]
        projected_points = (projection_matrix.T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
        projected_points_weights = 1. / (projected_points[:, -1:] + 1e-8)
        projected_points = (projected_points * projected_points_weights)[:, :-1]
         
        projected_points[:, :2] = ((projected_points[:, :2] + 1) * image_shape[1] - 1) / 2
        projected_points[:, 0] += w_begin
        projected_points[:, 1] += h_begin 
        drag_points.append(projected_points.mean(axis=0)) 

        if i == 0: 
            track_point_candidates = track_first(projected_points, (480, 720))            
            for j in range(tracks_init.shape[0]):
                x, y = tracks_init[j, 0], tracks_init[j, 1]
                target = np.array([x, y])
                candidate, track_point_candidates = find_and_remove_nearest_point(target, track_point_candidates)
                if candidate is not None:
                    track_idx.append(candidate[3].astype(np.int32))
                    mask[j] = True
                    
        tracks_output[0, i, mask] = projected_points[track_idx]
        tracks_output[0, i, ~mask, :2] = tracks_output[0, 0, ~mask, :2]
        tracks_output[0, i, ~mask, 2] = 2
    
    track_template.item()['tracks'] = tracks_output 
    track_template.item()['drag_points'] = np.stack(drag_points, axis=0)
    sub_dir = f'{output_dir}/tracks_gen'
    os.makedirs(sub_dir, exist_ok=True)
     
    np.save(f'{sub_dir}/tracks.npy', track_template)
    visualize_tracks(tracks_dir=sub_dir, output_dir=sub_dir, args=args)

def run_diffusion(args, output_dir): 
 
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(args.model_cfg_path)
    cfg = OmegaConf.merge(schema, cfg)
    n_training_frames = cfg.train_dataset.n_training_frames
    n_frames_interval = cfg.train_dataset.n_frames_interval
    norm_fac = cfg.train_dataset.norm_fac
    model = MDM_ST(cfg.pc_size, n_training_frames, n_feats=3, model_config=cfg.model_config).to(device)

    ckpt = load_file(args.model_path, device='cpu')
    model.load_state_dict(ckpt, strict=True)
    model.eval().requires_grad_(False)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)
    pipeline = TrajPipeline(model=model, scheduler=noise_scheduler)
 
    pc_path = f'{output_dir}/point_cloud.ply'  
    pc = trimesh.load_mesh(pc_path)
    points = pc.vertices
    points = np.array(points)  
    points, center, scale = transform2origin(points, size=1)
    np.save(f'{output_dir}/center.npy', center)
    np.save(f'{output_dir}/scale.npy', scale)

    N = 2048
    max_num_forces = 1
    has_gravity = args.mat_label > 0

    points = torch.tensor(points, dtype=torch.float32, device=device).contiguous()
    ratio_N = N / points.shape[0]
    idx = fps(points, ratio=ratio_N, random_start=True)
    np.save(f'{output_dir}/fps_idx.npy', idx.cpu().numpy())
    points_tensor = points[idx].contiguous()
    points_center = shift2center_th(points_tensor) # MPM coordinate
    points = points_tensor.cpu().numpy()

    # User input
    if "drag_mode" in cfg_json:
        if cfg_json["drag_mode"] == "point":
            drag_point = np.array(cfg_json["drag_point"])
        elif cfg_json["drag_mode"] == "max":
            drag_point_idx = np.argmax(points[:, cfg_json["drag_axis"]]) if cfg_json["drag_mode"] == "max" \
                else np.argmin(points[:, cfg_json["drag_axis"]])
            drag_point = points[drag_point_idx]
        else:
            raise ValueError(f"Invalid drag mode: {cfg_json['drag_mode']}")
        drag_offset = np.abs(points - drag_point)
        drag_mask = (drag_offset < 0.4).all(axis=-1)
        drag_dir = np.array(cfg_json["drag_dir"], dtype=np.float32)
        drag_dir /= np.linalg.norm(drag_dir)
        drag_force = drag_dir * np.array(cfg_json["force_coeff"])
    else:
        drag_mask = np.ones(N, dtype=bool)
        drag_point = np.zeros(4)
        drag_dir = np.zeros(3)
        drag_force = np.zeros(3) 
    
    if cfg_json["material"] == "elastic":
        log_E, nu = np.array(cfg_json["log_E"]), np.array(cfg_json["nu"])
    else: 
        log_E, nu = np.array(6), np.array(0.4) # Default values for non-elastic materials

    print(f'[Diffusion Simulation] Number of drag points: {drag_mask.sum()}/{N}')
    print(f'[Diffusion Simulation] Drag point: {drag_point}')
    print(f'[Diffusion Simulation] log_E: {log_E}, ν: {nu}')
    print(f'[Diffusion Simulation] Drag force: {drag_force}')
    print(f'[Diffusion Simulation] Material type: {cfg_json["material"]}({args.mat_label})')
    print(f'[Diffusion Simulation] Has gravity: {has_gravity}')

    force_order = torch.arange(max_num_forces) 
    mask = torch.from_numpy(drag_mask).bool()
    mask = mask.unsqueeze(0) if mask.ndim == 1 else mask  
     
    batch = {} 
    batch['gravity'] = torch.from_numpy(np.array(has_gravity)).long().unsqueeze(0)
    batch['drag_point'] = torch.from_numpy(drag_point).float() / 2
    batch['drag_point'] = batch['drag_point'].unsqueeze(0) # (1, 4)
    batch['points_src'] = points_tensor.float().unsqueeze(0) / 2 

    if has_gravity:
        floor_normal = np.load(f'{output_dir}/floor_normal.npy')
        floor_height = np.load(f'{output_dir}/floor_height.npy') * scale / 2.
        batch['floor_height'] = torch.from_numpy(np.array(floor_height)).float().unsqueeze(0)

        # Create rotation matrix to align floor normal with [0, 1, 0] (upward direction)
        target_normal = np.array([0, 1, 0])
        
        # Use Rodrigues' rotation formula to find rotation matrix
        # Rotate from floor_normal to target_normal
        v = np.cross(floor_normal, target_normal)
        s = np.linalg.norm(v)
        c = np.dot(floor_normal, target_normal)
        
        if s < 1e-6:  # If vectors are parallel
            if c > 0:  # Same direction
                R_floor = np.eye(3)
            else:  # Opposite direction
                R_floor = -np.eye(3)
        else:
            v = v / s
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_floor = np.eye(3) + s * K + (1 - c) * (K @ K)

        R_floor_tensor = torch.from_numpy(R_floor).float().to(device)
        for i in range(batch['points_src'].shape[0]):
            batch['points_src'][i] = (R_floor_tensor @ batch['points_src'][i].T).T
    else:
        batch['floor_height'] = torch.ones(1).float() * -2.4

    print(f'[Diffusion Simulation] Floor height: {batch["floor_height"]}')

    if mask.shape[1] == 0:
        mask = torch.zeros(0, N).bool()
        batch['force'] = torch.zeros(0, 3)
        batch['drag_point'] = torch.zeros(0, 4) 
    else:
        batch['force'] = torch.from_numpy(drag_force).float().unsqueeze(0)
     
    batch['mat_type'] = torch.from_numpy(np.array(args.mat_label)).long()
    if np.array(batch['mat_type']).item() == 3: # Rigid dataset
        batch['is_mpm'] = torch.tensor(0).bool()
    else:
        batch['is_mpm'] = torch.tensor(1).bool()
    
    if has_gravity: # Currently we only have either drag force or gravity  
        batch['force'] = torch.tensor([[0, -1.0, 0]]).to(device)   
    
    all_forces = torch.zeros(max_num_forces, 3)
    all_forces[:batch['force'].shape[0]] = batch['force']
    all_forces = all_forces[force_order]
    batch['force'] = all_forces

    all_drag_points = torch.zeros(max_num_forces, 4)  
    all_drag_points[:batch['drag_point'].shape[0], :batch['drag_point'].shape[1]] = batch['drag_point'] # The last dim of drag_point is not used now
    all_drag_points = all_drag_points[force_order]
    batch['drag_point'] = all_drag_points

    if batch['gravity'][0] == 1: # add gravity to force
        batch['force'] = torch.tensor([[0, -1.0, 0]]).float().to(device) 

    all_mask = torch.zeros(max_num_forces, N).bool()
    all_mask[:mask.shape[0]] = mask
    all_mask = all_mask[force_order]

    batch['mask'] = all_mask[..., None] # (n_forces, N, 1) for compatibility
    batch['E'] = torch.from_numpy(log_E).unsqueeze(-1).float() if log_E > 0 else torch.zeros(1).float()
    batch['nu'] = torch.from_numpy(nu).unsqueeze(-1).float()

    for k in batch:
        batch[k] = batch[k].unsqueeze(0).to(device)

     
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = pipeline(batch['points_src'], batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1],
            batch['drag_point'], batch['floor_height'], batch['gravity'], coeff=batch['E'], generator=torch.Generator().manual_seed(args.seed), 
            device=device, batch_size=1, y=batch['mat_type'], n_frames=n_training_frames, num_inference_steps=25)
        output = output.cpu().numpy()  
        for j in range(output.shape[0]):
            if batch['gravity'][0] == 1:
                for k in range(output.shape[1]):
                    output[j, k] = (np.linalg.inv(R_floor) @ output[j, k].T).T 
            np.save(f'{output_dir}/gen_data.npy', output[j:j+1].squeeze())

def run_vggt(args, output_dir):
 
    if not os.path.exists(f'{output_dir}/est_pcd.npy'):
        
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        if os.path.exists(f'{args.base_dir}/{args.data_name}/input_ori.png'):
            image_names = [f'{args.base_dir}/{args.data_name}/input_ori.png']
        else:
            image_names = [f'{args.base_dir}/{args.data_name}/input.png']
             
        images = []
        for image_name in image_names:
            image = Image.open(image_name) 
            image = np.array(image)[2:-2, 3:-3]
            image = image.astype(np.float32) / 255.0
            images.append(image)
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        images = images[:, :3]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(images)

        est_pcd = predictions['world_points'].cpu().numpy()
        depth = predictions['depth'].cpu().numpy() 
        Image.fromarray((depth[0, 0, :, :, 0] * 255).astype(np.uint8)).save(f'{output_dir}/est_depth.png')
        np.save(f'{output_dir}/est_pcd.npy', est_pcd)
        est_pcd_export = trimesh.PointCloud(est_pcd.reshape(-1, 3))
        est_pcd_export.export(f'{output_dir}/est_pcd.ply')

    cfg_json_path = f'{args.base_dir}/{args.data_name}/config.json'
    with open(cfg_json_path, 'r') as f:
        cfg_json = json.load(f)
    floor_loc_begin = np.array(cfg_json["floor_loc_begin"])
    floor_loc_end = np.array(cfg_json["floor_loc_end"])

    input_mask = np.array(Image.open(f'{output_dir}/input_mask.png').convert('L')) 
    input_mask_eroded = input_mask.copy()
    kernel = np.ones((5, 5), np.uint8) 
    input_mask_eroded = cv2.erode(input_mask_eroded, kernel, iterations=1)
    Image.fromarray(input_mask_eroded).save(f'{output_dir}/input_mask_eroded.png')

    est_pcd = np.load(f'{output_dir}/est_pcd.npy')[0, 0]
    est_pcd = np.pad(est_pcd, ((2, 2), (3, 3), (0, 0)), mode='constant', constant_values=0)
    est_pcd_masked = est_pcd[input_mask_eroded > 0].reshape(-1, 3)
    est_pcd_floor = est_pcd[floor_loc_begin[0]:floor_loc_end[0],
        floor_loc_begin[1]:floor_loc_end[1]].reshape(-1, 3)
    
    bmax = est_pcd_masked.max(axis=0)
    bmin = est_pcd_masked.min(axis=0)
    aabb = bmax - bmin
    center = (bmax + bmin) / 2
    scale = aabb.max()
    est_pcd = (est_pcd - center) / scale
    est_pcd_masked = (est_pcd_masked - center) / scale
    est_pcd_floor = (est_pcd_floor - center) / scale

    projection_matrix = np.load('templates/projection.npy')
    crop_info = np.load(f'{output_dir}/crop_info.npy')
    h_begin, w_begin, res = crop_info[0], crop_info[1], crop_info[2] 
    image_shape = (res, res)  # Example image shape (H, W)

    pc_path = f'{output_dir}/point_cloud.ply'
    pc = trimesh.load_mesh(pc_path)
    points = pc.vertices
    points = np.array(points) 

    projected_points = (projection_matrix.T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    projected_points_weights = 1. / (projected_points[:, -1:] + 1e-8)
    projected_points = (projected_points * projected_points_weights)[:, :-1]
        
    projected_points[:, :2] = ((projected_points[:, :2] + 1) * image_shape[1] - 1) / 2
    projected_points[:, 0] += w_begin
    projected_points[:, 1] += h_begin

    gt_pcd = np.zeros((480, 720, 3))
    min_z = np.ones((480, 720)) * 233
    for i, project_point in enumerate(projected_points):
        y, x = int(project_point[1]), int(project_point[0])
        if project_point[2] < min_z[y, x]:
            gt_pcd[y, x] = points[i]
            min_z[y, x] = project_point[2]

    gt_pcd_masked = gt_pcd[input_mask_eroded > 0] 
    min_z_masked = min_z[input_mask_eroded > 0]
    min_z_num = min_z_masked.shape[0]
    z_values_threshold = np.sort(min_z_masked)[min_z_num // 3] 

    est_pcd_masked_ori = est_pcd_masked.copy()
    est_pcd_masked = est_pcd_masked[min_z_masked < z_values_threshold]
    gt_pcd_masked = gt_pcd_masked[min_z_masked < z_values_threshold]

    est_pcd_masked_export = trimesh.PointCloud(est_pcd_masked)
    est_pcd_masked_export.export(f'{output_dir}/est_pcd_masked.ply')
    gt_pcd_masked_export = trimesh.PointCloud(gt_pcd_masked)
    gt_pcd_masked_export.export(f'{output_dir}/gt_pcd_masked.ply')    
    
    # Use least squares to find the best-fit similarity transformation (rotation + translation + scale)
    # between est_pcd_masked and gt_pcd_masked (correspondences are known and ordered)
    # This is an extension of the Kabsch algorithm to include scaling

    # Compute centroids
    est_centroid = np.mean(est_pcd_masked, axis=0)
    gt_centroid = np.mean(gt_pcd_masked, axis=0)

    # Center the point clouds
    est_centered = est_pcd_masked - est_centroid
    gt_centered = gt_pcd_masked - gt_centroid

    # Compute covariance matrix
    H = est_centered.T @ gt_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale factor 
    scale = np.trace(R.T @ H) / np.trace(est_centered.T @ est_centered)

    # Compute translation
    t = gt_centroid - scale * R @ est_centroid

    # Compose transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t 

    # Apply transformation
    est_pcd_masked_ori_transformed = scale * (R @ est_pcd_masked_ori.T).T + t
    est_pcd_transformed = scale * (R @ est_pcd_masked.T).T + t
    est_pcd_transformed_export = trimesh.PointCloud(est_pcd_transformed)
    est_pcd_transformed_export.export(f'{output_dir}/est_pcd_masked_transformed.ply')
    est_pcd_floor_transformed = scale * (R @ est_pcd_floor.T).T + t
    est_pcd_floor_transformed_export = trimesh.PointCloud(est_pcd_floor_transformed)
    est_pcd_floor_transformed_export.export(f'{output_dir}/est_pcd_floor_transformed.ply')

    # Compute RMSE for the alignment
    alignment_rmse = np.sqrt(np.mean(np.sum((est_pcd_transformed - gt_pcd_masked) ** 2, axis=1)))
 
    # Fit a plane using PCA to get normal vector and center point
    center = np.mean(est_pcd_floor_transformed, axis=0) 
    pca = PCA(n_components=3)
    pca.fit(est_pcd_floor_transformed)
    normal = pca.components_[2]  # Last component is normal to plane  

    # Calculate floor height as distance between the center of est_pcd_masked and the fitted floor plane 
    d = -np.dot(normal, center)  # d parameter for plane equation
    est_centroid = np.mean(est_pcd_masked_ori_transformed, axis=0)  # center of est_pcd_masked
    est_centroid[1] = 0 # set y to 0
    floor_height = np.abs(np.dot(est_centroid, normal) + d) / np.linalg.norm(normal) 

    print(f"[Floor Alignment] Floor Height: {-floor_height}")
    print(f"[Floor Alignment] Floor Normal: {normal}") 
    np.save(f'{output_dir}/floor_normal.npy', normal)
    np.save(f'{output_dir}/floor_height.npy', -floor_height)

def run_LGM(args, output_dir):
    
    device = torch.device("cuda")
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    sys.argv = ['pipeline_track_gen.py', 'big']
    opt = tyro.cli(AllConfigs)

    model = LGM(opt)
    ckpt = load_file(args.lgm_ckpt_path, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.half().to(device)
    model.eval()

    rays_embeddings = model.prepare_default_rays(device)
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    for i in range(4):
        image = Image.open(f"{output_dir}/view_{i}.png")
        image = image.resize((256, 256))
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        images.append(image)
    mv_image = np.stack(images, axis=0)
    
    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, f'{output_dir}/point_cloud.ply')

        # render front view
        cam_poses = torch.from_numpy(orbit_camera(0, 0, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        # cam_poses = torch.from_numpy(orbit_camera(45, 225, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]

        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
        image_save = (image[0, 0].permute(1, 2, 0).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(image_save).save(f'{output_dir}/front_view.png')

        images = []
        azimuth = np.arange(0, 360, 2, dtype=np.int32)
        elevation = 0
        
        for azi in tqdm(azimuth):
            
            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(f'{output_dir}/gs_animation.mp4', images, fps=30)
        
def run_sv3d(args, output_dir):
    
    model_path = "chenguolin/sv3d-diffusers"
    data_dir = f'{output_dir}/data'
    os.makedirs(data_dir, exist_ok=True)

    num_frames, sv3d_res = 20, 576
    elevations_deg = [args.elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    
    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(model_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")

    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, 
        unet=unet, vae=vae,
        scheduler=scheduler,
    )

    pipeline = pipeline.to("cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                
            image = Image.open(f'{output_dir}/input_processed.png')
            if len(image.split()) == 4:  # RGBA
                input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
                input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
            else:
                input_image = image
            
            video_frames = pipeline(
                input_image.resize((sv3d_res, sv3d_res)),
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,  # smaller to save memory
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=torch.manual_seed(args.seed),
            ).frames[0]

    torch.cuda.empty_cache()
    gc.collect()

    export_to_gif(video_frames, f"{output_dir}/view_animation.gif", fps=7)
    for i, frame in enumerate(video_frames): 
        frame.save(f"{data_dir}/{i:03d}.png")
    
    save_idx = [19, 4, 9, 14]
    for i in range(4):
        video_frames[save_idx[i]].save(f"{output_dir}/view_{i}.png")

def run_sam(args, output_dir):
    
    # Load SAM checkpoint
    sv3d_res = 576
    sam_predictor = sam_init(args.sam_ckpt_path)
    print("[SAM] Loaded SAM model")
    
    input_raw = Image.open(f'{args.base_dir}/{args.data_name}/input.png') if not os.path.exists(f'{args.base_dir}/{args.data_name}/input_masked.png') else Image.open(f'{args.base_dir}/{args.data_name}/input_masked.png')
    input_sam = sam_out_nosave(sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw))
    mask = np.array(input_sam)[:, :, 3]
    Image.fromarray(mask).save(f"{output_dir}/input_mask.png")
    y, x, res = image_preprocess(input_sam, f"{output_dir}/input_processed.png", target_res=sv3d_res,
        lower_contrast=False, rescale=True) 
    np.save(f"{output_dir}/crop_info.npy", np.array([y, x, res]))

from das.models.pipelines import DiffusionAsShaderPipeline
from das.infer import load_media
def run_das(args, output_dir, prompt, seed):
    output_dir = os.path.join(args.output_dir, args.data_name)
    das = DiffusionAsShaderPipeline(gpu_id=args.gpu, output_dir=os.path.join(args.output_dir, args.data_name))
    video_tensor, fps, is_video = load_media(f'{args.base_dir}/{args.data_name}/input.png')
    tracking_tensor, _, _ = load_media(os.path.join(args.output_dir, args.data_name, 'tracks_gen', 'tracking', 'tracks_tracking.mp4'))
    das.apply_tracking(
        video_tensor=video_tensor,
        fps=24,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=None,
        prompt=prompt,
        checkpoint_path=args.das_ckpt_path,
        seed=seed
    )
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../examples", type=str, help="Base dir")
    parser.add_argument("--output_dir", default="../outputs", type=str, help="Output filepath")
    parser.add_argument("--sam_ckpt_path", default="../checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--lgm_ckpt_path", default="../checkpoints/lgm_fp16.safetensors") 
    parser.add_argument("--das_ckpt_path", default="../checkpoints/cogshader5B")
    parser.add_argument("--base_ckpt_path", default="../checkpoints/physctrl_base.safetensors")
    parser.add_argument("--large_ckpt_path", default="../checkpoints/physctrl_large.safetensors")
    parser.add_argument("--gpu", type=int, default=0)    
    parser.add_argument("--data_name", default="chair", type=str, help="Data Name")
    parser.add_argument("--base_cfg_path", default="configs/eval_base.yaml", type=str, help="Model config")
    parser.add_argument("--large_cfg_path", default="configs/eval_large.yaml", type=str, help="Model config")
    parser.add_argument("--elevation", default=0, type=float, help="Camera elevation of the input image")
    parser.add_argument("--seed", default=0, type=int, help="Random seed") 
    parser.add_argument('--tracks_dir', type=str, default='', help='DAS Tracking data directory')
    parser.add_argument('--output_fps', type=int, default=24, help='DAS Output video FPS')
    parser.add_argument('--point_size', type=int, default=10, help='DAS Tracking point size')
    parser.add_argument('--len_track', type=int, default=0, help='DAS Tracking trajectory length')
    parser.add_argument('--num_frames', type=int, default=49, help='DAS Number of frames to generate black video')
    
    args = parser.parse_args() 
    seed_everything(args.seed) 
    mat_labels = {'elastic': 0, 'plasticine': 1, 'sand': 2, 'rigid': 3}

    output_dir = f'{args.output_dir}/{args.data_name}'
    cfg_json_path = f'{args.base_dir}/{args.data_name}/config.json'
    with open(cfg_json_path, 'r') as f:
        cfg_json = json.load(f)
    args.model_path = args.base_ckpt_path
    args.model_cfg_path = args.base_cfg_path

    mat_type = cfg_json['material']
    if mat_type in mat_labels:
        args.mat_label = mat_labels[mat_type]
    else:
        raise ValueError(f"Invalid material type: {mat_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    ## Run SAM to preprocess the input image
    run_sam(args, output_dir)

    ## Run SV3D to generate 21 frames
    run_sv3d(args, output_dir)
    
    ## Run LGM to reconstruct the 3D model
    run_LGM(args, output_dir)   
    
    ## Run VGGT to infer floor height and floor normal
    if args.mat_label > 0:
        args.model_path = args.large_ckpt_path
        args.model_cfg_path = args.large_cfg_path
        run_vggt(args, output_dir) 

    ## Run Generation to get results and tracks
    run_diffusion(args, output_dir) 
    run_track(args, output_dir) 
        
    ## Run Video Generation
    prompt = cfg_json['prompt']
    run_das(args, output_dir, prompt, seed=cfg_json['seed'] if 'seed' in cfg_json else 42)
    
    
    