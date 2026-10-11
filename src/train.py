import argparse
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from options import TrainingConfig

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from torchvision import transforms

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDPMPipeline, DDIMScheduler, DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, UNet2DModel
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_cosine_schedule_with_warmup
from pipeline_traj import TrajPipeline
from accelerate.utils import DistributedDataParallelKwargs

from model.spacetime import MDM_ST
from dataset.traj_dataset import TrajDataset

from utils.visualization import save_pointcloud_video, save_pointcloud_json, save_threejs_html
from utils.physics import loss_momentum
from utils.physics import DeformLoss

logger = get_logger(__name__)

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main(args):
    vis_dir = os.path.join(args.output_dir, args.vis_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = {}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

        src_snapshot_folder = os.path.join(cfg.output_dir, 'src')
        ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
        for folder in ['model', 'dataset']:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        shutil.copy(os.path.abspath(__file__), os.path.join(cfg.output_dir, 'src', 'train.py'))

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    model = MDM_ST(args.pc_size, args.train_dataset.n_training_frames, n_feats=3, model_config=args.model_config)

    # if args.gradient_checkpointing:
    #     model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params = model.parameters()
    # Optimizer creation
    optimizer = optimizer_class(
        [
            {"params": params, "lr": args.learning_rate},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # if args.model_type == 'dit_st_water':
    #     from dataset.water_dataset import TrajDataset
    # Dataset and DataLoaders creation:
    train_dataset = TrajDataset('train', args.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, pin_memory=True)

    val_dataset = TrajDataset('val', args.train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    # noise = torch.randn(sample_image.shape)
    # timesteps = torch.LongTensor([50])
    # noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
    # Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num of Trainable Parameters (M) = {sum(p.numel() for p in model.parameters()) / 1000000}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Log to = {args.output_dir}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    loss_deform = DeformLoss()

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, (batch, _) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                latents = batch['points_tgt'] # (bsz, n_frames, n_points, 3)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.condition_drop_rate > 0:
                    # Randomly drop some of the latents
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    null_emb = (random_p > args.condition_drop_rate).float()[..., None, None]
                else:
                    null_emb = None

                # Predict the noise residual
                pred_sample = model(noisy_latents, timesteps, batch['points_src'], batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1], batch['drag_point'], batch['floor_height'], batch['gravity'], batch['base_drag_coeff'], y=None if 'mat_type' not in batch else batch['mat_type'], null_emb=null_emb)
                losses = {}

                loss = F.mse_loss(pred_sample.float(), latents.float())
                losses['xyz'] = loss.detach().item()

                if args.lambda_mask > 0:
                    loss_mask = F.mse_loss(pred_sample[batch['mask']], latents[batch['mask']])
                    loss += loss_mask
                    losses['mask'] = loss_mask.detach().item()

                if args.lambda_vel > 0.:
                    target_vel = latents[:, 1:] - latents[:, :-1]
                    pred_vel = (pred_sample[:, 1:] - pred_sample[:, :-1])
                    loss_vel = F.mse_loss(target_vel.float(), pred_vel.float())
                    losses['loss_vel'] = loss_vel.detach().item()
                    loss = loss + loss_vel

                if 'vol' in batch and args.lambda_momentum > 0.:
                    loss_p = loss_momentum(x=pred_sample, vol=batch['vol'], force=batch['weighted_force'],
                        drag_pt_num=batch['mask'][:, 0, :].sum(dim=1), norm_fac=args.train_dataset.norm_fac)
                    losses['loss_p'] = loss_p.detach().item()
                    loss = loss + args.lambda_momentum * loss_p
                
                if 'vol' in batch and args.lambda_deform > 0.:
                    pred_sample_mpm = pred_sample
                    if 'is_mpm' in batch:
                        mask = batch['is_mpm']
                        pred_sample_mpm = pred_sample[mask]
                        batch['vol'] = batch['vol'][mask]
                        batch['F'] = batch['F'][mask]
                        batch['C'] = batch['C'][mask]
                    loss_F = loss_deform(x=pred_sample_mpm.clamp(min=-2.2, max=2.2), vol=batch['vol'], F=batch['F'],
                        C=batch['C'], frame_interval=2, norm_fac=args.train_dataset.norm_fac) if batch['vol'].shape[0] > 0 else torch.tensor(0.0, device=pred_sample.device)
                    losses['loss_deform'] = loss_F.detach().item()
                    loss = loss + args.lambda_deform * loss_F

                if args.model_config.floor_cond:
                    floor_height = batch['floor_height'].reshape(bsz, 1, 1) # (B, 1, 1)
                    sample_min_height = torch.amin(latents[..., 1], dim=(1, 2)).reshape(bsz, 1, 1)
                    floor_height = torch.minimum(floor_height, sample_min_height)
                    loss_floor = (torch.relu(floor_height - pred_sample[..., 1]) ** 2).mean()
                    losses['loss_floor'] = loss_floor.detach().item()
                    loss += loss_floor

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
                if global_step % cfg.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        model.eval()
                        pipeline = TrajPipeline(model=accelerator.unwrap_model(model), scheduler=DDIMScheduler.from_config(noise_scheduler.config))
                        logger.info(
                            f"Running validation... \n."
                        )
                        for i, (batch, _) in enumerate(val_dataloader):
                            with torch.autocast("cuda"):
                                gs = [1.0] if args.condition_drop_rate == 0 else [1.0, 2.0, 3.0]
                                for guidance_scale in gs:
                                    output = pipeline(batch['points_src'], batch['force'], batch['E'], batch['nu'], batch['mask'][..., :1], batch['drag_point'], batch['floor_height'], batch['gravity'], batch['base_drag_coeff'], y=None if 'mat_type' not in batch else batch['mat_type'], device=accelerator.device, batch_size=args.eval_batch_size, generator=torch.Generator().manual_seed(args.seed), n_frames=args.train_dataset.n_training_frames, guidance_scale=guidance_scale)
                                    output = output.cpu().numpy()
                                    tgt = batch['points_tgt'].cpu().numpy()
                                    save_dir = os.path.join(vis_dir, f'{global_step:06d}')
                                    os.makedirs(save_dir, exist_ok=True)
                                    for j in range(output.shape[0]):
                                        save_pointcloud_video(output[j:j+1].squeeze(), tgt[j:j+1].squeeze(), os.path.join(save_dir, f'{i*batch["points_src"].shape[0] + j}_{guidance_scale}.gif'),
                                            drag_mask=batch['mask'][j:j+1, 0, :, 0].cpu().numpy().squeeze(), vis_flag=args.train_dataset.dataset_path)
                                        # pred_name = f'{i*batch["points_src"].shape[0]+j}_pred.json'
                                        # gt_name = f'{i*batch["points_src"].shape[0]+j}_gt.json'
                                        # save_pointcloud_json(output[j:j+1].squeeze(), os.path.join(save_dir, pred_name))
                                        # save_pointcloud_json(tgt[j:j+1].squeeze(), os.path.join(save_dir, gt_name))
                                        # save_threejs_html(pred_name, gt_name, os.path.join(save_dir, f'{j}.html'))
                                torch.cuda.empty_cache()
                        model.train()

            logs = losses
            logs.update({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = unet.to(torch.float32)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(schema, cfg)
    main(cfg)