import numpy as np
import torch
import gc
from PIL import Image
import sys
import os

# Add the project root directory to Python path (use absolute paths for robustness)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "libs"))
sys.path.append(os.path.join(project_root, "libs", "LGM"))
sys.path.append(os.path.join(project_root, "libs", "das"))
sys.path.append(os.path.join(project_root, "src"))

from sv3d.diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler, DDIMScheduler
from diffusers.utils import export_to_gif, export_to_video
from kiui.cam import orbit_camera
from safetensors.torch import load_file
from omegaconf import OmegaConf

from LGM.core.models import LGM
from LGM.core.options import AllConfigs 
from LGM.core.gs import GaussianRenderer
from .track_utils.visualize_tracks import visualize_tracks
from .track_utils.preprocessing import track_first, find_and_remove_nearest_point
from .interpolate import interpolate_points
from das.models.pipelines import DiffusionAsShaderPipeline

import h5py
import tyro
from tqdm import tqdm
from options import TestingConfig
from pipeline_traj import TrajPipeline
from model.spacetime import MDM_ST
from argparse import Namespace

def load_sv3d_pipeline(device, model_path="chenguolin/sv3d-diffusers"):
    unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(model_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
    pipeline = StableVideo3DDiffusionPipeline(
        image_encoder=image_encoder, feature_extractor=feature_extractor, 
        unet=unet, vae=vae,
        scheduler=scheduler,
    ).to(device)
    return pipeline

def load_LGM(opt, device, lgm_ckpt_path="./checkpoints/lgm_fp16.safetensors"):
    model = LGM(opt)
    ckpt = load_file(lgm_ckpt_path, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.half().to(device)
    model.eval()
    return model

def load_diffusion(device, model_cfg_path, diffusion_ckpt_path, seed=0):
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(model_cfg_path)
    cfg = OmegaConf.merge(schema, cfg)
    n_training_frames = cfg.train_dataset.n_training_frames
    n_frames_interval = cfg.train_dataset.n_frames_interval
    norm_fac = cfg.train_dataset.norm_fac

    model = MDM_ST(cfg.pc_size, n_training_frames, n_feats=3, model_config=cfg.model_config).to(device)

    ckpt = load_file(diffusion_ckpt_path, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval().requires_grad_(False)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)
    pipeline = TrajPipeline(model=model, scheduler=noise_scheduler)
    return pipeline

def gen_tracking_video(base_dir):
    
    animated_points = np.load(f'{base_dir}/gen_data.npy')
    animated_points = animated_points * 2
    new_animate_points = np.zeros((49, 2048, 3))
    for i in range(47):
        if i % 2 == 0:  
            new_animate_points[i + 1] = animated_points[i // 2]
        else:
            new_animate_points[i + 1] = (animated_points[i // 2] + animated_points[i // 2 + 1]) / 2
    new_animate_points[0] = new_animate_points[1]
    new_animate_points[48] = new_animate_points[47]
    animated_points = new_animate_points

    projection_matrix = np.load(f'{base_dir}/projection.npy')
    crop_info = np.load(f'{base_dir}/crop_info.npy')
    center = np.load(f'{base_dir}/center.npy')
    scale = np.load(f'{base_dir}/scale.npy')
    animated_points = (animated_points / scale) + center    

    ## Aligned to Gaussian points at this moment
    print(animated_points.mean(), animated_points.std(), animated_points.max(), animated_points.min())
    device = torch.device("cuda")
    sys.argv = ['pipeline_track_gen.py', 'big']
    opt = tyro.cli(AllConfigs)

    scale_factor = 2
    focal = 0.5 * opt.output_size / np.tan(np.deg2rad(opt.fovy) / 2)
    new_fovy_rad = scale_factor * np.arctan(opt.output_size / focal)
    new_fovy_deg = np.rad2deg(new_fovy_rad)
    opt.fovy = new_fovy_deg
    opt.output_size *= scale_factor # Expand canvas size by 2

    gs = GaussianRenderer(opt)
    gaussians = gs.load_ply(f'{base_dir}/point_cloud.ply', compatible=True).to(device).float()
    idx = torch.from_numpy(np.load(f'{base_dir}/idx.npy')).to(device)
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

    for i in tqdm(range(0, 49, 1)):
        drive_current = torch.from_numpy(animated_points[i]).to(device).float()
        ret_points, new_rotation = interpolate_points(gaussian_pos, gaussians[:, 7:11], drive_x, drive_current, topk_index)
        gaussians_new = gaussians.clone()
        gaussians_new[:, :3] = ret_points
        gaussians_new[:, 7:11] = new_rotation
        pos.append(ret_points.cpu().numpy())

        # with torch.no_grad():
        #     ret = gs.render(gaussians_new.unsqueeze(0), cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)
        #     mask = (ret['alpha'][0,0].permute(1, 2, 0).contiguous().float().cpu().numpy() * 255.0).astype(np.uint8)
        #     image = (ret['image'][0, 0].permute(1, 2, 0).contiguous().float().cpu().numpy()*255.0).astype(np.uint8)
        #     image_save = np.concatenate([image, mask], axis=-1)

        #     h_begin, w_begin, res = crop_info[0], crop_info[1], crop_info[2]
        #     h_begin = h_begin - (256 * scale_factor - 256)
        #     w_begin = w_begin - (256 * scale_factor - 256)
        #     image_save = Image.fromarray(image_save).resize((res * scale_factor, res * scale_factor), Image.LANCZOS) 
    
    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates', 'tracks_template.npy')
    track_template = np.load(template_path, allow_pickle=True)
    tracks = track_template.item()['tracks']
    tracks_output = tracks.copy()
    tracks_init = tracks[0, 0]  
    track_idx = []
    mask = np.zeros(tracks_init.shape[0], dtype=bool)
    
    for i in tqdm(range(49)):
        
        # points = animated_points[i]
        points = pos[i]
    
        projected_points = (projection_matrix.T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
        projected_points_weights = 1. / (projected_points[:, -1:] + 1e-8)
        projected_points = (projected_points * projected_points_weights)[:, :-1]
        
        h_begin, w_begin, res = crop_info[0], crop_info[1], crop_info[2]
        image_shape = (res, res)  # Example image shape (H, W)
        projected_points[:, :2] = ((projected_points[:, :2] + 1) * image_shape[1] - 1) / 2
        projected_points[:, 0] += w_begin
        projected_points[:, 1] += h_begin

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
    # track_template.item()['drag_points'] = np.stack(drag_points, axis=0)
    sub_name = 'tracks_gen'
    sub_dir = f'{base_dir}/{sub_name}'
    os.makedirs(sub_dir, exist_ok=True)

    np.save(f'{sub_dir}/tracks.npy', track_template)
    args = Namespace(tracks_dir=sub_dir, output_dir=sub_dir, output_fps=24, point_size=10, len_track=0, num_frames=49, video_path=None)
    visualize_tracks(tracks_dir=sub_dir, output_dir=sub_dir, args=args)

def load_das(gpu_id, output_dir):
    das = DiffusionAsShaderPipeline(gpu_id=gpu_id, output_dir=output_dir)
    return das

def normalize_points(output_dir, fluid=False):
    from .transform import transform2origin, shift2center
    import trimesh
    from torch_cluster import fps
    
    device = 'cuda'
    
    pc_path = f'{output_dir}/point_cloud.ply'
    pc = trimesh.load_mesh(pc_path)
    points = pc.vertices
    points = np.array(points)
    points, center, scale = transform2origin(points, size=1)
    N = 2048
    grid_center = [5, 5, 5]
    drag_size = [0.4, 0.4, 0.4]

    points = shift2center(points, center=grid_center)
    points = torch.tensor(points, dtype=torch.float32, device=device).contiguous()
    np.save(f'{output_dir}/center.npy', center)
    np.save(f'{output_dir}/scale.npy', scale)
    ratio_N = N / points.shape[0]
    idx = fps(points, ratio=ratio_N, random_start=True)
    points = points[idx].cpu().numpy()
    np.save(f'{output_dir}/idx.npy', idx.cpu().numpy())
    return points, center, scale