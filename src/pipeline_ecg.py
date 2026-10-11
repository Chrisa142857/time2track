"""
Inference pipeline for ECG-conditioned heart mesh trajectory generation.

Usage:
    pipeline = ECGTrajPipeline(model=model, scheduler=DDIMScheduler(...))
    trajectory = pipeline(
        init_pc        = rest_mesh,       # (B, 1, N, 3)
        ecg            = ecg_signal,      # (B, 12, 600)
        device         = 'cuda',
        batch_size     = B,
        n_frames       = 49,
        num_inference_steps = 50,
        guidance_scale = 1.0,
        generator      = torch.Generator().manual_seed(0),
    )
    # trajectory: (B, n_frames, N, 3)
"""

import torch
from diffusers import DiffusionPipeline


class ECGTrajPipeline(DiffusionPipeline):
    """DDIM/DDPM sampling pipeline for ECGCondMDM."""

    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        init_pc:             torch.Tensor,   # (B, 1, N, 3) or (B, N, 3)
        ecg:                 torch.Tensor,   # (B, 12, 600)
        device,
        batch_size:          int   = 1,
        num_inference_steps: int   = 50,
        guidance_scale:      float = 1.0,
        n_frames:            int   = 49,
        generator:           torch.Generator = None,
        delay_mat:           torch.Tensor = None,   # (B, 12, 12), optional
    ) -> torch.Tensor:

        n_points = init_pc.shape[-2]

        # ── Initialise noisy trajectory ──────────────────────────────────
        sample = torch.randn(
            (batch_size, n_frames, n_points, 3),
            generator=generator,
        ).to(device)

        self.model.to(device)
        init_pc = init_pc.to(device)
        ecg     = ecg.to(device)
        if delay_mat is not None:
            delay_mat = delay_mat.to(device)

        do_cfg = (guidance_scale > 1.0)

        # For classifier-free guidance: concatenate conditioned & unconditioned
        null_emb_ones  = torch.ones(batch_size,  1, 1, device=device, dtype=sample.dtype)
        null_emb_zeros = torch.zeros(batch_size, 1, 1, device=device, dtype=sample.dtype)

        if do_cfg:
            init_pc_in   = torch.cat([init_pc] * 2)
            ecg_in       = torch.cat([ecg]     * 2)
            delay_mat_in = torch.cat([delay_mat] * 2) if delay_mat is not None else None
            null_emb     = torch.cat([null_emb_zeros, null_emb_ones])
        else:
            init_pc_in   = init_pc
            ecg_in       = ecg
            delay_mat_in = delay_mat
            null_emb     = null_emb_ones

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.progress_bar(self.scheduler.timesteps):
            t_batch = torch.tensor([t] * batch_size, device=device)

            if do_cfg:
                sample_in = torch.cat([sample] * 2)
                t_in      = torch.cat([t_batch] * 2)
            else:
                sample_in = sample
                t_in      = t_batch

            model_output = self.model(
                sample_in, t_in, init_pc_in, ecg_in,
                null_emb=null_emb, delay_mat=delay_mat_in,
            )

            if do_cfg:
                pred_uncond, pred_cond = model_output.chunk(2)
                model_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            sample = self.scheduler.step(model_output, t_batch[0], sample).prev_sample

        return sample
