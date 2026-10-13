"""
Training script for ECG-conditioned 4-D heart mesh trajectory generation.

Adapts the PhysCtrl training loop (train.py) to use:
  - UKbiobankHeartMesh dataset  (ECG + heart mesh sequences)
  - ECGCondMDM model            (ECG tokens replace force/physics conditioning)
  - ECGTrajPipeline             (DDIM sampling for validation)

Launch via:
    accelerate launch --config_file configs/acc/1gpu.yaml \
        src/train_ecg_mesh.py --config src/configs/config_ecg_heart.yaml
"""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import is_wandb_available
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import diffusers
from diffusers import DDIMScheduler, DDPMScheduler  # now valid
from dataset.ukbiobank_dataset import UKbiobankHeartMesh
from model.ecg_mesh_model import ECGCondMDM
from pipeline_ecg import ECGTrajPipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    acc_project = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=acc_project,
        kwargs_handlers=[ddp_kwargs],
    )
        
    if args.report_to == 'wandb':
        if not is_wandb_available():
            raise ImportError('Install wandb for W&B logging.')
        import wandb  # noqa: F401

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config={})

    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # ── Output directories & config snapshot ────────────────────────────
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        OmegaConf.save(args, os.path.join(args.output_dir, 'config.yaml'))

        src_snapshot = os.path.join(args.output_dir, 'src')
        ignore_fn = lambda d, files: [f for f in files if '__pycache__' in f]
        for folder in ['model', 'dataset']:
            shutil.copytree(folder, os.path.join(src_snapshot, folder),
                            ignore=ignore_fn, dirs_exist_ok=True)
        shutil.copy(os.path.abspath(__file__),
                    os.path.join(src_snapshot, 'train_ecg_mesh.py'))

    # ── Model ────────────────────────────────────────────────────────────
    n_points = args.dataset.n_points        # 388 for UK Biobank heart
    n_frames = args.dataset.n_training_frames  # e.g. 49

    model = ECGCondMDM(
        n_points=n_points,
        n_frames=n_frames,
        n_feats=3,
        model_config=args.model_config,
    )

    # ── Warm start: load model weights only from an existing accelerate
    # checkpoint dir, leaving optimizer + LR schedule fresh.  Unlike
    # `resume_from_checkpoint` (which must live inside output_dir and also
    # restores the end-of-cosine LR ≈ 0), this lets a converged checkpoint be
    # fine-tuned into a NEW output_dir with a real learning rate.
    warm = getattr(args, 'warm_start_from', None)
    if warm:
        from safetensors.torch import load_file
        _sd = load_file(os.path.join(warm, 'model.safetensors'))
        # ref_mode='full' widens delay_mlp.0 from max_level to n_leads*max_level
        # columns. Scatter the single-reference weights into THEIR reference's column
        # block and zero every other reference, so the full-map model is numerically
        # IDENTICAL to the shipped single-reference one at step 0. Any later difference
        # is then attributable to the added references and to nothing else -- the same
        # zero-init discipline as the alpha_delay gate.
        _own = model.state_dict()
        for _k, _v in list(_sd.items()):
            if not _k.endswith('delay_mlp.0.weight') or _k not in _own:
                continue
            _tgt = _own[_k]
            if _tgt.shape == _v.shape:
                continue
            _enc = model.ecg_encoder
            _L, _r = int(_enc.max_level), int(_enc.ref_lead)
            assert _tgt.shape[1] == _enc.n_leads * _L and _v.shape[1] == _L, \
                f'{_k}: cannot map {tuple(_v.shape)} -> {tuple(_tgt.shape)}'
            _w = torch.zeros_like(_tgt)
            _w[:, _r * _L:(_r + 1) * _L] = _v.to(_w.dtype)
            _sd[_k] = _w
            logger.info(f'[warm-start] widened {_k} {tuple(_v.shape)} -> '
                        f'{tuple(_tgt.shape)}; reference {_r} block seeded, rest zero')
        missing, unexpected = model.load_state_dict(_sd, strict=False)
        logger.info(f'[warm-start] from {warm}: '
                    f'missing={list(missing)} unexpected={list(unexpected)}')
        assert not missing and not unexpected, 'warm-start state_dict mismatch'

    # ── Progress readout: has the delay branch actually re-converged? ──
    # Snapshot the warm-start weights so we can report how far each branch has
    # travelled.  Without this a short fine-tune is indistinguishable from a
    # long one and "metrics unchanged" is unfalsifiable.
    # substring split, not a registry: branch names are stable here.
    _DELAY_KEYS = ('alpha_delay', 'beta_raw', 'delay_mlp', 'lead_pos_embed', 'delay_block')
    _init_w = {n: p.detach().clone().cpu().float()
               for n, p in model.named_parameters() if 'ecg_encoder' in n}

    def _branch_drift():
        """(delay ‖Δθ‖/‖θ0‖, morph ‖Δθ‖/‖θ0‖) since warm start."""
        acc = {'delay': [0.0, 0.0], 'morph': [0.0, 0.0]}
        for n, p in accelerator.unwrap_model(model).named_parameters():
            if n not in _init_w:
                continue
            k = 'delay' if any(t in n for t in _DELAY_KEYS) else 'morph'
            d = p.detach().cpu().float() - _init_w[n]
            acc[k][0] += float((d * d).sum())
            acc[k][1] += float((_init_w[n] ** 2).sum())
        return tuple((a / max(b, 1e-12)) ** 0.5 for a, b in acc.values())

    def _alpha_delay():
        enc = getattr(accelerator.unwrap_model(model), 'ecg_encoder', None)
        a = getattr(enc, 'alpha_delay', None)
        return float(a.detach()) if a is not None else float('nan')

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Optimizer ────────────────────────────────────────────────────────
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    train_dataset = UKbiobankHeartMesh('train', args.dataset)
    val_dataset   = UKbiobankHeartMesh('val',   args.dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────
    num_warmup = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        steps_per_epoch = math.ceil(
            math.ceil(len(train_dataloader) / accelerator.num_processes)
            / args.gradient_accumulation_steps
        )
        num_training_steps = args.num_train_epochs * steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )

    # ── Accelerator prepare ──────────────────────────────────────────────
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    # ── Logging ──────────────────────────────────────────────────────────
    total_batch = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info('***** Running ECG-conditioned heart mesh training *****')
    logger.info(f'  Trainable parameters (M) = '
                f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}')
    logger.info(f'  Train subjects = {len(train_dataset)}')
    logger.info(f'  Train batches / epoch = {len(train_dataloader)}')
    logger.info(f'  Epochs = {args.num_train_epochs}')
    logger.info(f'  Batch size per device = {args.train_batch_size}')
    logger.info(f'  Total batch size = {total_batch}')
    logger.info(f'  Gradient accumulation = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimisation steps = {args.max_train_steps}')
    logger.info(f'  Output dir = {args.output_dir}')

    global_step        = 0
    first_epoch        = 0
    best_val_loss      = float('inf')
    best_val_error     = float('inf')
    last_val_loss      = float('nan')
    last_val_error     = float('nan')
    early_stop_counter = 0

    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == 'latest':
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint')],
                key=lambda x: int(x.split('-')[1])
            )
            path = dirs[-1] if dirs else None
        else:
            path = os.path.basename(args.resume_from_checkpoint)

        if path is None:
            accelerator.print('No checkpoint found. Starting fresh.')
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f'Resuming from {path}')
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step         = int(path.split('-')[1])
            initial_global_step = global_step
            first_epoch         = global_step // steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc='Steps',
        disable=not accelerator.is_local_main_process,
    )

    # ── Diffusion scheduler ───────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type='sample',
        clip_sample=False,
    )

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed is not None else None
    )

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()

        for step, (batch, _) in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                latents = batch['points_tgt']   # (B, T, N, 3)
                bsz     = latents.shape[0]

                # Forward diffusion: add noise
                noise          = torch.randn_like(latents)
                timesteps      = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                noisy_latents  = noise_scheduler.add_noise(latents, noise, timesteps)

                # Classifier-free guidance: randomly null-out ECG conditioning
                if args.condition_drop_rate > 0:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # 1 = keep conditioning, 0 = drop (null)
                    null_emb = (random_p >= args.condition_drop_rate).float()[:, None, None]
                else:
                    null_emb = None

                # Denoising prediction
                delay_mat = batch.get('delay_mat')
                pred_sample, aux = model(
                    noisy_latents,
                    timesteps,
                    batch['points_src'],
                    batch['ecg'],
                    null_emb=null_emb,
                    delay_mat=delay_mat,
                    return_aux=True,
                )

                losses = {}

                # Primary MSE loss
                loss = F.mse_loss(pred_sample.float(), latents.float())
                losses['loss_xyz'] = loss.detach().item()

                # Velocity consistency loss
                if args.lambda_vel > 0.0:
                    target_vel = latents[:, 1:] - latents[:, :-1]
                    pred_vel   = pred_sample[:, 1:] - pred_sample[:, :-1]
                    loss_vel   = F.mse_loss(pred_vel.float(), target_vel.float())
                    loss       = loss + args.lambda_vel * loss_vel
                    losses['loss_vel'] = loss_vel.detach().item()

                # Encoder auxiliary losses (populated only by localized_delay encoder)
                for aux_name, aux_val in aux.items():
                    weight = getattr(args, f'lambda_{aux_name}', 0.0)
                    if weight > 0.0:
                        loss = loss + weight * aux_val
                        losses[f'loss_{aux_name}'] = aux_val.detach().item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Post-step bookkeeping ────────────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        _prune_checkpoints(args)
                        save_path = os.path.join(
                            args.output_dir, f'checkpoint-{global_step}'
                        )
                        accelerator.save_state(save_path)
                        logger.info(f'Saved checkpoint to {save_path}')

                # Validation
                if global_step % args.validation_steps == 0 or global_step == 1:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        val_loss, val_error = _run_validation(
                            model, noise_scheduler, val_dataloader, accelerator,
                            args, global_step, generator,
                        )
                        accelerator.log(
                            {'val_loss': val_loss, 'val_error': val_error},
                            step=global_step,
                        )
                        _dd, _md = _branch_drift()
                        _a = _alpha_delay()
                        logger.info(
                            f'Step {global_step}: val_loss={val_loss:.6f}  '
                            f'val_error={val_error:.6f}'
                        )
                        logger.info(
                            f'step={global_step} alpha_delay={_a:.6e} '
                            f'delay_drift={_dd:.4e} morph_drift={_md:.4e} '
                            f'val_loss={val_loss:.6f} val_error={val_error:.6f} '
                            f'lr={lr_scheduler.get_last_lr()[0]:.3e}'
                        )
                        accelerator.log({'alpha_delay': _a,
                                         'delay_drift': _dd,
                                         'morph_drift': _md}, step=global_step)
                        last_val_loss  = val_loss
                        last_val_error = val_error
                        if val_loss < best_val_loss:
                            best_val_loss      = val_loss
                        # if val_error < best_val_error:
                        #     best_val_error      = val_error
                            early_stop_counter = 0
                            best_path = os.path.join(
                                args.output_dir,
                                f'best_val-loss{val_loss:.6f}_step{global_step}')
                            accelerator.save_state(best_path)
                            _point_best_model(args.output_dir, best_path)
                            logger.info(
                                f'New best val_loss={best_val_loss:.6f} – '
                                f'New best val_error={best_val_error:.6f} – '
                                f'saved to {best_path}'
                            )
                        else:
                            early_stop_counter += 1
                            logger.info(
                                f'No improvement ({early_stop_counter}/'
                                f'{args.early_stop_patience})'
                            )
                        model.train()

            logs = losses
            logs.update({
                'loss':        loss.detach().item(),
                'lr':          lr_scheduler.get_last_lr()[0],
                'val_loss':    last_val_loss,
                'val_error':   last_val_error,
                'alpha_delay': _alpha_delay(),   # dense gate trajectory
            })
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # Early stopping check (after each epoch's steps)
        patience = getattr(args, 'early_stop_patience', 3)
        if patience > 0 and early_stop_counter >= patience:
            logger.info(
                f'Early stopping triggered after {patience} validations '
                f'without improvement.'
            )
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


# ---------------------------------------------------------------------------
# Helpers called from main
# ---------------------------------------------------------------------------

def _point_best_model(output_dir, best_path):
    """`best_model` is a link to the metric-named best checkpoint: eval scripts keep their
    default while the checkpoint itself carries its val loss and step."""
    link = os.path.join(output_dir, 'best_model')
    if os.path.islink(link):
        old = os.path.realpath(link)
        os.remove(link)
        # keep one best, as the former in-place overwrite of best_model did
        if (os.path.dirname(old) == os.path.realpath(output_dir) and old != os.path.realpath(best_path)
                and os.path.basename(old).startswith('best_val-loss')):
            shutil.rmtree(old)
    elif os.path.isdir(link):
        shutil.rmtree(link)          # pre-link layout: the same replacement the old code made
    os.symlink(os.path.basename(best_path), link)


def _prune_checkpoints(args):
    if args.checkpoints_total_limit is None:
        return
    checkpoints = sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint')],
        key=lambda x: int(x.split('-')[1])
    )
    if len(checkpoints) >= args.checkpoints_total_limit:
        n_remove = len(checkpoints) - args.checkpoints_total_limit + 1
        for ck in checkpoints[:n_remove]:
            shutil.rmtree(os.path.join(args.output_dir, ck))
            logger.info(f'Removed old checkpoint {ck}')


@torch.no_grad()
def _run_validation(model, noise_scheduler, val_dataloader, accelerator,
                    args, global_step, generator):
    """Estimate val diffusion loss and mean per-vertex L2 error (Chamfer proxy).

    Returns
    -------
    val_loss  : float  – average DDPM denoising MSE on val set
    val_error : float  – average per-vertex L2 distance between predicted and
                         ground-truth trajectories (metres / mesh units)
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    pipeline = ECGTrajPipeline(
        model=unwrapped,
        scheduler=DDIMScheduler.from_config(noise_scheduler.config),
    )
    logger.info(f'Running validation at step {global_step} …')

    # Use the guidance scale that matches training CFG setting
    gs = 2.0 if args.condition_drop_rate > 0 else 1.0

    val_loss_sum  = 0.0
    val_error_sum = 0.0
    n_samples     = 0

    # ── Diffusion (denoising) loss on val set ────────────────────────────
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type='sample',
        clip_sample=False,
    )
    gen_loss = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed or 0)
    )
    for i, (batch, _) in enumerate(val_dataloader):
        if i >= args.num_val_batches:
            break
        latents = batch['points_tgt'].to(accelerator.device)   # (B, T, N, 3)
        bsz     = latents.shape[0]
        noise   = torch.randn_like(latents)
        timesteps = torch.randint(
            0, ddpm_scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device, generator=gen_loss,
        ).long()
        noisy_latents = ddpm_scheduler.add_noise(latents, noise, timesteps)

        delay_mat = batch.get('delay_mat')
        if delay_mat is not None:
            delay_mat = delay_mat.to(accelerator.device)
        with torch.autocast('cuda'):
            pred = unwrapped(
                noisy_latents,
                timesteps,
                batch['points_src'].to(accelerator.device),
                batch['ecg'].to(accelerator.device),
                null_emb=None,
                delay_mat=delay_mat,
            )
        val_loss_sum += F.mse_loss(pred.float(), latents.float()).item() * bsz
        n_samples    += bsz

    val_loss = val_loss_sum / max(n_samples, 1)

    # ── Trajectory error: per-vertex L2 (mean over T, N, xyz) ───────────
    n_samples = 0
    error_sum = 0.0
    for i, (batch, _) in enumerate(val_dataloader):
        if i >= args.num_val_batches:
            break
        bsz = batch['points_src'].shape[0]
        delay_mat = batch.get('delay_mat')
        if delay_mat is not None:
            delay_mat = delay_mat.to(accelerator.device)
        with torch.autocast('cuda'):
            traj = pipeline(
                init_pc=batch['points_src'].to(accelerator.device),
                ecg=batch['ecg'].to(accelerator.device),
                device=accelerator.device,
                batch_size=bsz,
                n_frames=args.dataset.n_training_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=gs,
                generator=torch.Generator().manual_seed(args.seed or 0),
                delay_mat=delay_mat,
            )                                                   # (B, T, N, 3)
        gt    = batch['points_tgt'].to(accelerator.device)     # (B, T, N, 3)
        # mean per-vertex L2 distance over all frames and vertices
        error = (traj.float() - gt.float()).pow(2).sum(-1).sqrt().mean()
        error_sum += error.item() * bsz
        n_samples += bsz

        torch.cuda.empty_cache()

    val_error = error_sum / max(n_samples, 1)
    return val_loss, val_error


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cli_args = parser.parse_args()

    # Load config directly – ECGTrainingConfig is a superset of TrainingConfig
    # and not all fields exist in the shared schema, so we skip the strict merge.
    cfg = OmegaConf.load(cli_args.config)
    # Allow attribute-style access and resolve interpolations
    OmegaConf.set_struct(cfg, False)

    main(cfg)
