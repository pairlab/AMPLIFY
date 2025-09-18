import math
import os
import sys
from datetime import datetime

import hydra
import numpy as np
import pyinstrument
import torch
from IPython.core import ultratb
from omegaconf import OmegaConf
from torch.backends import opt_einsum
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import wandb
from amplify.models.motion_tokenizer import MotionTokenizer
from amplify.utils.cfg_utils import merge_checkpoint_config, get_device
from amplify.utils.data_utils import points_to_velocities, rel_cls_logits_to_diffs, velocities_to_points
from amplify.utils.logger import Logger
from amplify.utils.metrics import get_normalized_codebook_perplexity, get_traj_metrics
from amplify.utils.train import (
    DummyGradScaler,
    get_checkpoint_dir,
    get_dataloaders,
    get_datasets,
    get_root_dir,
    latest_checkpoint_from_dir,
    load_checkpoint,
    save_checkpoint,
)
from amplify.utils.vis_utils import vis_pred

# sys.excepthook = ultratb.FormattedTB(mode='Plain', color_scheme='Linux', call_pdb=1)

torch.multiprocessing.set_start_method('spawn', force=True)
opt_einsum.strategy = 'auto-hq' # seems to speed up training by 10% or so

def train_epoch(train_global_iter, model, train_loader, optimizer, scaler, device, logger, cfg):
    model.train()
    grad_accum = math.ceil(cfg.batch_size / cfg.gpu_max_bs) # number of gradient accumulations
    train_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        x = batch['traj'].to(device)

        with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=cfg.amp):
            x_gt = points_to_velocities(x, time_dim=2)

            if cfg.cond_on_img:
                img = batch['images'].to(device)
                x_recon, codebook_indices, rel_logits = model(x_gt, img)
            else:
                x_recon, codebook_indices, rel_logits = model(x_gt)

            # Compute loss
            loss = model.get_loss(x_recon=x_recon, rel_logits=rel_logits, gt_vel=x_gt, gt_traj=x)

        # Backprop
        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum == 0:
            if cfg.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.detach()

        x_recon = rel_cls_logits_to_diffs(
                logits=rel_logits,
                pred_views=len(cfg.cond_cameraviews),
                num_tracks=cfg.num_tracks,
                rel_cls_img_size=cfg.loss.rel_cls_img_size,
                global_img_size=cfg.img_shape,
                get_last_timestep=False
        )

        x_recon_vis = velocities_to_points(x_recon, time_dim=2, init_points=x[:, :, [0]])

        # Logging
        metrics = get_traj_metrics(x_recon_vis, x, cfg.img_shape)
        logger.update({"train_loss": loss.item(), "train_metrics": metrics}, train_global_iter, phase='train')

        if train_global_iter % cfg.log_interval == 0:
            gt_img = vis_pred(batch['images'][[0]], x[[0]].cpu()).numpy()
            vis_img = vis_pred(batch['images'][[0]], x_recon_vis[[0]].cpu()).numpy()
            combined_img = np.concatenate([gt_img[0], vis_img[0]], axis=0)
            codebook_perplexity = get_normalized_codebook_perplexity(codebook_indices, cfg.codebook_size)
            logger.log({"train_reconstructions": [wandb.Image(combined_img)], "train_codebook_perplexity": codebook_perplexity}, train_global_iter, flatten=False)

        train_global_iter += cfg.gpu_max_bs

    avg_train_loss = train_loss.item() / len(train_loader)

    return avg_train_loss, train_global_iter

@torch.no_grad()
def val_epoch(val_global_iter, model, val_loader, device, logger, cfg):
    model.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
        x = batch['traj'].to(device)

        with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=cfg.amp):
            x_gt = points_to_velocities(x, time_dim=2)

            if cfg.cond_on_img:
                img = batch['images'].to(device)
                x_recon, codebook_indices, rel_logits = model(x_gt, img)
            else:
                x_recon, codebook_indices, rel_logits = model(x_gt)

            # Compute loss, metrics
            loss = model.get_loss(x_recon=x_recon, rel_logits=rel_logits, gt_vel=x_gt, gt_traj=x)

        val_loss += loss.detach()

        x_recon = rel_cls_logits_to_diffs(
                logits=rel_logits,
                pred_views=len(cfg.cond_cameraviews),
                num_tracks=cfg.num_tracks,
                rel_cls_img_size=cfg.loss.rel_cls_img_size,
                global_img_size=cfg.img_shape,
                get_last_timestep=False
        )

        x_recon_vis = velocities_to_points(x_recon, time_dim=2, init_points=x[:, :, [0]])
        codebook_perplexity = get_normalized_codebook_perplexity(codebook_indices, cfg.codebook_size)

        # Logging
        metrics = get_traj_metrics(x_recon_vis, x, cfg.img_shape)
        logger.update({"val_loss": loss.item(), "val_metrics": metrics, "val_codebook_perplexity": codebook_perplexity.item()}, val_global_iter, phase='val')

        if val_global_iter % cfg.log_interval == 0:
            gt_img = vis_pred(batch['images'][[0]], x[[0]].cpu()).numpy()
            vis_img = vis_pred(batch['images'][[0]], x_recon_vis[[0]].cpu()).numpy()
            combined_img = np.concatenate([gt_img[0], vis_img[0]], axis=0)
            # plt.imshow(combined_img)
            # plt.show()
            logger.log({"val_reconstructions": [wandb.Image(combined_img)]}, val_global_iter, phase='val',flatten=False)

        val_global_iter += cfg.gpu_max_bs

    avg_val_loss = val_loss.item() / len(val_loader)

    return avg_val_loss, val_global_iter


@hydra.main(config_path="cfg", config_name="train_motion_tokenizer", version_base="1.2")
def main(cfg):
    run_name = str(cfg.run_name) or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cfg.checkpoint is not None:
        cfg = merge_checkpoint_config(cfg)
    print("================== FINAL CONFIG ==================")
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    print("Using device: ", device)
    if cfg.profile:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        profiler = pyinstrument.Profiler()

    if cfg.root_dir is None:
        root_dir = get_root_dir()
    else:
        root_dir = cfg.root_dir

    # Dataloaders
    train_datasets, val_datasets = get_datasets(
        root_dir=root_dir,
        train_datasets=cfg.train_datasets,
        val_datasets=cfg.val_datasets,
        keys_to_load=cfg.keys_to_load,
        motion_tokenizer_cfg=cfg,
        task_names=cfg.task_names,
    )

    train_dataloader_dict, val_dataloader_dict = get_dataloaders(
        train_datasets,
        val_datasets,
        gpu_max_bs=cfg.gpu_max_bs,
        num_workers=cfg.num_workers,
        quick=cfg.quick,
    )
    train_loader = train_dataloader_dict['traj']

    if val_dataloader_dict is not None:
        val_loader = val_dataloader_dict['traj']
    else:
        val_loader = None

    # Model
    model = MotionTokenizer(cfg).to(device)
    if cfg.compile:
        model = torch.compile(model)

    # Optimizer, Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.adam_betas)
    warmup_epochs = cfg.num_epochs // 10
    if cfg.lr_schedule == 'cosine':
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01,total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs,eta_min=cfg.lr / 100)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    elif cfg.lr_schedule is None:
        scheduler = None
    else:
        raise ValueError(f"Invalid lr_schedule: {cfg.lr_schedule}")

    if cfg.amp:
        scaler = torch.amp.GradScaler(enabled=cfg.amp)
    else:
        scaler = DummyGradScaler()

    # Checkpoint dir
    resume = (cfg.checkpoint is not None or cfg.resume)
    checkpoint_dir = get_checkpoint_dir(stage="motion_tokenizer", run_name=run_name, resume=resume)
    print(f"Checkpoint dir: {checkpoint_dir}")
    if cfg.resume and cfg.checkpoint is None:
        cfg.checkpoint = latest_checkpoint_from_dir(checkpoint_dir)

    # Load checkpoint
    if cfg.checkpoint is not None:
        model, optimizer, scheduler, scaler, checkpoint_cfg, checkpoint_info = load_checkpoint(cfg.checkpoint, model, optimizer, scheduler, scaler)
        scaler = DummyGradScaler() if scaler is None else scaler

        # run info
        start_epoch = checkpoint_info['epoch'] + 1
        end_epoch = cfg.num_epochs + 1
        train_global_iter = checkpoint_info['train_global_iter']
        val_global_iter = checkpoint_info['val_global_iter']
        wandb_run_id = checkpoint_info['wandb_run_id']
    else:
        start_epoch = 1
        end_epoch = cfg.num_epochs + 1
        train_global_iter = 0
        val_global_iter = 0
        wandb_run_id = None

    # Logging
    logger = Logger(train_log_interval=cfg.log_interval, val_log_interval=cfg.log_interval)
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.wandb_init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        name=run_name,
        config=wandb_cfg,
        mode='disabled' if not cfg.use_wandb else 'online',
        id=wandb_run_id,
        resume="allow" if cfg.checkpoint is not None else None,
        allow_val_change=True,
    )
    wandb.config.update({"checkpoint_dir":  checkpoint_dir}, allow_val_change=True)

    if 'SLURM_JOBID' in os.environ:
        wandb.config.update({'slurm_job_id': os.environ['SLURM_JOBID']}, allow_val_change=True)

    # Train
    for epoch in range(start_epoch, end_epoch):
        if cfg.profile:
            profiler.start()

        train_loss, train_global_iter = train_epoch(train_global_iter, model, train_loader, optimizer, scaler, device, logger, cfg)

        if val_loader is not None:
            val_loss, val_global_iter = val_epoch(val_global_iter, model, val_loader, device, logger, cfg)
        else:
            val_loss = 0
            val_global_iter = 0

        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.lr

        # Log
        print(f'Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | LR: {current_lr}')
        logger.log({"epoch": epoch, "learning_rate": current_lr, "avg_train_loss": train_loss, "avg_val_loss": val_loss}, train_global_iter)

        # Save checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        save_checkpoint(latest_path, epoch, cfg, model, optimizer, scaler, train_loss=train_loss, val_loss=val_loss, train_global_iter=train_global_iter, val_global_iter=val_global_iter, scheduler=scheduler)

        if epoch % cfg.save_interval == 0 and epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}.pt")
            save_checkpoint(checkpoint_path, epoch, cfg, model, optimizer, scaler, train_loss=train_loss, val_loss=val_loss, train_global_iter=train_global_iter, val_global_iter=val_global_iter, scheduler=scheduler)

        if cfg.profile:
            profiler.stop()
            profiler.print()


if __name__=='__main__':
    main()
