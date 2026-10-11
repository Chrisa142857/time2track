import torch
from diffusers import DDPMScheduler, DDIMScheduler
from dataset.traj_dataset import TrajDataset
from model.mdm import MDM
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
import torch.nn as nn
import torch.nn.functional as F
from eval import create_model
from tqdm import tqdm
import numpy as np
from utils.visualization import save_pointcloud_video, save_pointcloud_json, save_threejs_html
import matplotlib.pyplot as plt

def fibonacci_sphere(n):
    i   = torch.arange(n, dtype=torch.float32)
    phi = 2 * torch.pi * i / ((1 + 5**0.5) / 2)   # golden‑angle
    z   = 1 - 2 * (i + 0.5) / n                   # uniform in [-1,1]
    r_xy = (1 - z**2).sqrt()
    x = r_xy * torch.cos(phi)
    y = r_xy * torch.sin(phi)
    return torch.stack((x, y, z), dim=1)          # shape (n,3)

class Inferrer:
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.model = create_model(args).to(device)
        
        ckpt = load_file(args.resume, device='cpu')
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval().requires_grad_(False).to(device)
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)
        self.pipeline = TrajPipeline(model=self.model, scheduler=self.scheduler)

    @torch.no_grad()
    def probe_params(self, init_pc, force, motion_obs, mask, drag_point, floor_height, coeff, y, vis_dir=None, fname=None):
        out = []
        for e in torch.arange(4.0, 7.1, 0.5):
            # for n in torch.arange(0.2, 0.45, 0.05):
            # for n in [0.36]:
                E, nu = torch.tensor([e], device=self.device).reshape(1, 1), torch.tensor([n], device=self.device).reshape(1, 1)
                motion_pred = self.pipeline(init_pc, force, E, nu, mask, drag_point, floor_height, gravity=None, coeff=coeff, y=y, device=self.device, batch_size=1, generator=torch.Generator().manual_seed(self.args.seed), n_frames=self.args.train_dataset.n_training_frames, num_inference_steps=25)
                loss = F.mse_loss(motion_pred, motion_obs.to(self.device))
                out.append([loss, e, n])
                # save_pointcloud_video(motion_pred.squeeze().cpu().numpy(), motion_obs.squeeze().cpu().numpy(), os.path.join(f'{e.item():03f}_{nu.item():02f}.gif'), drag_mask=mask[:1, 0, :, 0].cpu().numpy().squeeze(), vis_flag='objaverse')
        out = torch.tensor(out).cpu().numpy()
        print("Best E, nu: ", out[np.argmin(out[:, 0])])
        plt.plot(out[:, 1], out[:, 0], marker='o', linestyle='-', linewidth=2)
        plt.xlabel('E')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(vis_dir, f'{fname}.png'))
        plt.close()
        
        return out

    def forward_model(self, motion_noisy, t, init_pc, force, E, nu, mask, guidance_scale=1.0):
        bsz = motion_noisy.shape[0]
        null_emb = torch.tensor([1] * motion_noisy.shape[0]).to(motion_noisy.dtype)
        if cfg > 1.0:
            motion_noisy = torch.cat([motion_noisy] * 2)
            init_pc = torch.cat([init_pc] * 2)
            force = torch.cat([force] * 2)
            E = torch.cat([E] * 2)
            nu = torch.cat([nu] * 2)
            t = torch.cat([t] * 2)
            mask = torch.cat([mask] * 2)
            null_emb = torch.cat([torch.tensor([0] * bsz).to(motion_noisy.dtype), null_emb])
        null_emb = null_emb[:, None, None].to(self.device, dtype=motion_noisy.dtype)
        model_output = self.model(motion_noisy, t, init_pc, force, E, nu, mask)
        if cfg > 1.0:
            model_pred_uncond, model_pred_cond = model_output.chunk(2)
            model_output = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
        return model_output

    def inference_model(self, init_pc, force, E, nu, mask, drag_point, floor_height, coeff,
        generator, 
        device, 
        batch_size: int = 1, 
        num_inference_steps: int = 50, 
        guidance_scale=1.0, 
        n_frames=20
    ):
        # Sample gaussian noise to begin loop
        sample = torch.randn((batch_size, n_frames, init_pc.shape[2], 3), generator=generator).to(device)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        do_classifier_free_guidance = (guidance_scale > 1.0)
        null_emb = torch.tensor([1] * batch_size).to(sample.dtype)
        if do_classifier_free_guidance:
            init_pc = torch.cat([init_pc] * 2)
            force = torch.cat([force] * 2)
            E = torch.cat([E] * 2)
            nu = torch.cat([nu] * 2)
            mask = torch.cat([mask] * 2)
            drag_point = torch.cat([drag_point] * 2)
            floor_height = torch.cat([floor_height] * 2)
            null_emb = torch.cat([torch.tensor([0] * batch_size).to(sample.dtype), null_emb])
        null_emb = null_emb[:, None, None].to(device)
        for t in self.scheduler.timesteps:
            t = torch.tensor([t] * batch_size, device=device)
            sample_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample
            t = torch.cat([t] * 2) if do_classifier_free_guidance else t
            # 1. predict noise model_output
            model_output = self.model(sample_input, t, init_pc, force, E, nu, mask, drag_point, floor_height, coeff, y=y, null_emb=null_emb)
            if do_classifier_free_guidance:
                model_pred_uncond, model_pred_cond = model_output.chunk(2)
                model_output = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
            sample = self.scheduler.step(model_output, t[0], sample).prev_sample
        return sample

    def estimate_params(self, model_name, motion_obs, init_pc, force, mask, drag_point, floor_height, coeff, y, cfg=1.0, gravity=None, probe=False, num_steps=400):
        device = 'cuda'
        
        all_loss = []
        if probe:
            out = []
            for e in torch.arange(4.0, 7.1, 0.5):
                E = torch.tensor([e], device=self.device).reshape(1, 1)
                motion_pred = self.pipeline(init_pc, force, E, nu, mask, drag_point, floor_height, gravity=gravity, coeff=coeff, y=y, device=self.device, batch_size=1, generator=torch.Generator().manual_seed(self.args.seed), n_frames=self.args.train_dataset.n_training_frames, num_inference_steps=25)
                loss = F.mse_loss(motion_pred, motion_obs.to(self.device))
                out.append([loss.item(), E.item()])
            out = torch.tensor(out)
            print("Best E, nu: ", out[torch.argmin(out[:, 0])])
            E = nn.Parameter(torch.tensor([out[np.argmin(out[:, 0]), 1]], device=device).reshape(1, 1))
            all_loss.append(out[torch.argmin(out[:, 0])])
        else:
            E = nn.Parameter(torch.tensor([4.5], device=device).reshape(1, 1))
        # nu = nn.Parameter(torch.tensor([0.15], device=device).reshape(1, 1))
        # force = nn.Parameter(torch.zeros([1, 3], device=device))
        # drag_point = nn.Parameter(torch.zeros([1, 3], device=device))
        optimizer = torch.optim.Adam([
            {'params': E, 'lr': 1e-2, 'min': 4.0, 'max': 7.0},
            # {'params': nu, 'lr': 1e-2, 'min': 0.15, 'max': 0.4},
            # {'params': [force, drag_point], 'lr': 1e-2}
        ])
        self.model.requires_grad_(True)
        progress_bar = tqdm(total=num_steps)
        progress_bar.set_description("Training")
        Es = []
        for step in range(num_steps):
            optimizer.zero_grad()
            noise = torch.randn_like(motion_obs, device=device)
            t = torch.randint(0, self.scheduler.num_train_timesteps, (motion_obs.shape[0],), device=device)
            motion_noisy = self.scheduler.add_noise(motion_obs, noise, t)
            model_output = self.model(motion_noisy, t, init_pc, force, E, nu, mask, drag_point, floor_height, gravity, coeff, y=y)
            loss = F.mse_loss(model_output, motion_obs)
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item(), 'E': E.item(), 'nu': nu.item()})
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                E.clamp_(4.0, 7.0)
            
            if (step + 1) % 200 == 0:
                Es.append(E.item())

            if (step + 1) % 200 == 0:
                motion_pred = self.pipeline(init_pc, force.detach(), E.detach(), nu.detach(), mask, drag_point.detach(), floor_height, gravity=gravity, coeff=coeff, y=y, device=self.device, batch_size=1, generator=torch.Generator().manual_seed(self.args.seed), n_frames=self.args.train_dataset.n_training_frames, num_inference_steps=25)
                loss = F.mse_loss(motion_pred, motion_obs)
                all_loss.append(torch.tensor([loss.item(), E.item()]))
        out = torch.stack(all_loss)
        print(out)
        E = out[torch.argmin(out[:, 0]), 1].to(device)
        Es.append(E.item())
        save_pointcloud_video(motion_pred.squeeze().cpu().numpy(), motion_obs.squeeze().cpu().numpy(), os.path.join(f'./debug/v3', f'{model_name}_{E.item():03f}_{nu.item():02f}.gif'), drag_mask=mask[:1, 0, :, 0].cpu().numpy().squeeze(), vis_flag='objaverse')
        return Es, nu, force, drag_point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(args.config)
    args = OmegaConf.merge(schema, cfg)

    val_dataset = TrajDataset('val', args.train_dataset)
    # val_dataset = [val_dataset[i] for i in range(len(val_dataset) - 15, len(val_dataset))]
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    inferrer = Inferrer(args)
    loss = 0.0
    loss_mask = 0.0
    probe = True
    num_steps = 400
    for i, (batch, _) in enumerate(val_dataloader):
        device = torch.device('cuda')
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_name = batch['model'][0]
            motion_obs = batch['points_tgt'].to(device)
            init_pc = batch['points_src'].to(device)
            force = batch['force'].to(device)
            E = batch['E'].to(device)
            nu = batch['nu'].to(device)
            mask = batch['mask'][..., :1].to(device, dtype=force.dtype)
            drag_point = batch['drag_point'].to(device)
            floor_height = batch['floor_height'].to(device)
            coeff = batch['base_drag_coeff']
            y=None if 'mat_type' not in batch else batch['mat_type'].to(device)
            gravity = batch['gravity'].to(device) if 'gravity' in batch else None
            print(model_name, floor_height)
            # for j in range(output.shape[0]):
            # save_pointcloud_video(motion_obs.squeeze().cpu().numpy(), motion_obs.squeeze().cpu().numpy(), os.path.join('./debug', f'{i:03d}_{E.item():03f}_{nu.item():02f}.gif'), drag_mask=mask[:1, 0, :, 0].cpu().numpy().squeeze(), vis_flag='objaverse')

            print('GT', E, nu, drag_point, force, y)

            est_E, est_nu, est_f, est_d = inferrer.estimate_params(model_name, motion_obs.to(device), init_pc, force, mask, drag_point, floor_height, coeff, y=y, cfg=1.0, gravity=gravity, probe=probe, num_steps=num_steps)
            # print(f'EST_{model_name}', F.mse_loss(est_E, E), F.mse_loss(est_nu, nu), F.mse_loss(est_d[..., :3], drag_point[..., :3]), F.mse_loss(est_f, force))
            est_E = ','.join([f'{e:.3f}' for e in est_E]) if isinstance(est_E, list) else est_E.item()
            print(est_E)
            with open(os.path.join('./debug', f'output_probe{probe}_steps{num_steps}.txt'), 'a+') as f:
                f.write(f'{model_name},{E.item()},{est_E},{nu.item()},{est_nu.item()},{drag_point.cpu().numpy()},{est_d.cpu().numpy()},{force.cpu().numpy()},{est_f.cpu().numpy()}\n')
        # break