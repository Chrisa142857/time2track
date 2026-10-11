import torch
from diffusers import DiffusionPipeline


class TrajPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, init_pc, force, E, nu, mask, drag_point, floor_height, gravity, coeff,
        generator, 
        device, 
        y = None,
        batch_size: int = 1, 
        num_inference_steps: int = 50, 
        guidance_scale=1.0, 
        n_frames=20
    ):
        # Sample gaussian noise to begin loop
        sample = torch.randn((batch_size, n_frames, init_pc.shape[2], 3), generator=generator).to(device)
        self.model.to(device)
        init_pc = init_pc.to(device)
        force = force.to(device)
        E = E.to(device)
        nu = nu.to(device)
        mask = mask.to(device).to(dtype=sample.dtype)
        drag_point = drag_point.to(device)
        floor_height = floor_height.to(device)
        coeff = coeff.to(device)
        gravity = gravity.to(device) if gravity is not None else None
        y = y.to(device) if y is not None else None
        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=device)
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
        for t in self.progress_bar(self.scheduler.timesteps):
            t = torch.tensor([t] * batch_size, device=device)
            sample_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample
            t = torch.cat([t] * 2) if do_classifier_free_guidance else t
            # 1. predict noise model_output
            model_output = self.model(sample_input, t, init_pc, force, E, nu, mask, drag_point, floor_height=floor_height, gravity_label=gravity, coeff=coeff, y=y, null_emb=null_emb)
            if do_classifier_free_guidance:
                model_pred_uncond, model_pred_cond = model_output.chunk(2)
                model_output = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1 
            sample = self.scheduler.step(model_output, t[0], sample).prev_sample
        return sample