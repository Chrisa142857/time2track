
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg

import kiui
import cv2
import numpy as np 

from kiui.cam import orbit_camera

from core.process_image import recenter
from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline
from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Ball(11,1) Plane(0,3)
torch.random.manual_seed(11)
np.random.seed(11)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# load image dream
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()

# process function
def process(opt: Options, path, task_name):
    
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8')

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    h_begin, w_begin, res = 50, 160, 380
    # h_begin, w_begin, res = -120, 0, 720
    image = recenter(carved_image, h_begin, w_begin, res)
    
    # generate mv
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:    
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    
    # image_out = Image.fromarray((image * 255).astype(np.uint8))
    # image_out = image_out.resize((256, 256), Image.BILINEAR)
    # image_out.save(os.path.join(opt.workspace, name + '_input.png'))
    # image = np.array(image_out).astype(np.float32) / 255.0
    # mv_image = pipe('A basketball', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    # mv_image = np.stack([image, mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    # mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], image], axis=0) # [4, 256, 256, 3], float32
    # mv_image = np.stack([mv_image[0], mv_image[2], mv_image[1], mv_image[3]], axis=0) # [4, 256, 256, 3], float32
    # for i in range(4):
    #     Image.fromarray((mv_image[i] * 255).astype(np.uint8)).save(os.path.join(opt.workspace, name + f'_{i}.png'))
    
    images = []
    for i in range(4):
        image = imageio.imread(os.path.join(opt.workspace, name + f'_{i}.png'))
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
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

        # render front view
        cam_poses = torch.from_numpy(orbit_camera(0, 0, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        
        os.makedirs(f'../../data/{task_name}_rendered', exist_ok=True)
        np.save(f'../../data/{task_name}_rendered/projection.npy', cam_view_proj[0].cpu().numpy())
        
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
        image_save = (image[0, 0].permute(1, 2, 0).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(image_save).save(os.path.join(opt.workspace, name + '_front_view.png'))
        
        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]
                scale = min(azi / 360, 1)
                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, name + '.mp4'), images, fps=30)

assert opt.test_path is not None
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]
    
task_name = 'chair'
for path in file_paths:
    if not task_name in path:
        continue
    process(opt, path, task_name=task_name)
