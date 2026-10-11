from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class TrainingConfig:
    image_size: int
    # train_batch_size = 16
    # eval_batch_size = 16  # how many images to sample during evaluation
    # num_epochs = 50
    # gradient_accumulation_steps = 1
    # learning_rate = 1e-4
    # lr_warmup_steps = 500
    # save_image_epochs = 10
    # save_model_epochs = 30
    # mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    # output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    # logging
    output_dir: str
    logging_dir: str
    vis_dir: str
    report_to: Optional[str]
    local_rank: int
    tracker_project_name: str

    # Training
    seed: Optional[int]
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: int
    max_train_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    use_8bit_adam: bool
    allow_tf32: bool
    dataloader_num_workers: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: Optional[float]
    prediction_type: Optional[str]
    mixed_precision: Optional[str]
    checkpointing_steps: int
    checkpoints_total_limit: Optional[int]
    resume_from_checkpoint: Optional[str]
    enable_xformers_memory_efficient_attention: bool
    validation_steps: int
    validation_train_steps: int
    validation_sanity_check: bool
    resume_step: Optional[int]
    push_to_hub: bool
    set_grads_to_none: bool
    lambda_vel: float
    lambda_mask : float
    lambda_momentum: float
    lambda_deform: float
    overfit: bool

    # Diffusion Specific
    condition_drop_rate: float

    # Dataset
    train_dataset: Dict

    # Model
    model_type: str
    pred_offset: bool
    model_config: Dict
    pc_size: int

@dataclass
class TestingConfig:
    dataloader_num_workers: int 
    pc_size: int
    model_type: str
    pred_offset: bool
    model_config: Dict
    train_dataset: Dict
    resume: str
    vis_dir: str
    eval_batch_size: int
    seed: int
    num_inference_steps: int