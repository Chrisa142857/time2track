from diffusers import DDPMScheduler, DDIMScheduler
from dataset.traj_dataset import TrajDataset
from model.mdm_dit import MDM_DiT
from model.spacetime import MDM_ST
import sys
from options import TrainingConfig, TestingConfig
from omegaconf import OmegaConf
from pipeline_traj import TrajPipeline
import torch
from safetensors.torch import load_file
import argparse
import os
import numpy as np
from utils.physics import loss_momentum, DeformLoss
import torch.nn.functional as F
from tqdm import tqdm
from utils.visualization import save_pointcloud_video, save_pointcloud_json, save_threejs_html, generate_html_from_exts

def create_model(args):
    model = MDM_ST(args.pc_size, args.train_dataset.n_training_frames, n_feats=3, model_config=args.model_config)
    return model

loss_deform = DeformLoss().to('cuda')
def main(args):
    val_dataset = TrajDataset('val', args.train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    device = 'cuda'
    model = create_model(args).to(device)
    ckpt = load_file(args.resume, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval().requires_grad_(False)
    model = torch.compile(model)
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)
    pipeline = TrajPipeline(model=model, scheduler=noise_scheduler)

    total_loss_p = 0.0
    total_loss_xyz = 0.0
    total_loss_F = 0.0
    total_loss_F_gt = 0.0
    for i, (batch, _) in enumerate(tqdm(val_dataloader)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipeline(batch['points_src'], batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1], batch['drag_point'], batch['floor_height'], batch['gravity'], batch['base_drag_coeff'], y=None if args.model_config.get('num_mat', 0) == 0 else batch['mat_type'], device=device, batch_size=args.eval_batch_size, generator=torch.Generator().manual_seed(args.seed), n_frames=args.train_dataset.n_training_frames, num_inference_steps=args.num_inference_steps)
            if 'vol' in batch:
                loss_F = loss_deform(x=output.clamp(min=-2.2, max=2.2), vol=batch['vol'].to(device), F=batch['F'].to(device),
                        C=batch['C'].to(device), frame_interval=2, norm_fac=args.train_dataset.norm_fac)
                loss_F_gt = loss_deform(x=batch['points_tgt'].to(device), vol=batch['vol'].to(device), F=batch['F'].to(device),
                        C=batch['C'].to(device), frame_interval=2, norm_fac=args.train_dataset.norm_fac)
                total_loss_F += loss_F
                total_loss_F_gt += loss_F_gt
            total_loss_xyz += F.mse_loss(output, batch['points_tgt'].to(device))
            output = output.cpu().numpy()
            tgt = batch['points_tgt'].cpu().numpy()
            vis_dir = args.vis_dir
            save_dir = os.path.join(vis_dir, f'test_100_{args.num_inference_steps}steps_nips_debug')
            os.makedirs(save_dir, exist_ok=True)
            for j in range(output.shape[0]):
                save_pointcloud_video(output[j:j+1].squeeze(), tgt[j:j+1].squeeze(), os.path.join(save_dir, f'{i*batch["points_src"].shape[0] + j:03d}_{batch["E"][j].item():03f}_{batch["nu"][j].item():03f}.gif'), drag_mask=batch['mask'][j:j+1, 0, :, 0].cpu().numpy().squeeze(), vis_flag='objaverse')
                np.save(os.path.join(save_dir, f'{i*batch["points_src"].shape[0] + j}_{batch["E"][j].item():03f}_{batch["nu"][j].item():03f}.npy'), output[j:j+1].squeeze())
                np.save(os.path.join(save_dir, f'{batch["model"][j]}.npy'), output[j:j+1].squeeze())
            torch.cuda.empty_cache()
    generate_html_from_exts(save_dir, os.path.join(save_dir, f'visualize.html'), 'gif')
    print(total_loss_p, total_loss_xyz, total_loss_F, total_loss_F_gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)