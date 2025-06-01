#!/usr/bin/env python
# coding=utf-8
# Modified FLUX training script with DPO loss support

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
import io
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

# Import datasets library for loading DPO dataset
from datasets import load_dataset, load_from_disk

#if is_wandb_available():
#    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.34.0.dev0")

logger = get_logger(__name__)


# Add EMA implementation
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model: model to apply EMA
            decay: decay factor
        """
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
        
    def to(self, device):
        """
        Move the shadow parameters to the specified device
        """
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)
        return self


class DPODataset(Dataset):
    """Dataset for DPO training with FLUX"""
    
    def __init__(
        self,
        dataset_path,
        size=1024,
        center_crop=True,
        flip_flag=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.flip_flag = flip_flag
         
        # Load the dataset created by your DPO dataset creation script
        if os.path.isdir(dataset_path):
            self.dataset = load_from_disk(dataset_path)
        else:
            # Assume it's a HuggingFace dataset
            self.dataset = load_dataset(dataset_path)
        
        # Use train split
        if 'train' in self.dataset:
            self.dataset = self.dataset['train']
        
        # Setup transforms
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=0.5)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.train_resize = train_resize
        self.train_crop = train_crop
        self.train_flip = train_flip
        self.train_transforms = train_transforms
       
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load images from bytes
        preferred_image = Image.open(io.BytesIO(item['jpg_0']))
        less_preferred_image = Image.open(io.BytesIO(item['jpg_1']))
        
        # Process images
        images = []
        for image in [preferred_image, less_preferred_image]:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = self.train_resize(image)
            
            # Random horizontal flip (apply same flip to both images)
            #self.flip_flag = random.random() < 0.5
            
            #if self.flip_flag:
            #    image = self.train_flip(image)
            
            # Crop (apply same crop to both images)
            if self.center_crop:
                image = self.train_crop(image)
            else:
                self.crop_params = self.train_crop.get_params(image, (self.size, self.size))
                y1, x1, h, w = self.crop_params
                image = crop(image, y1, x1, h, w)
            
            image = self.train_transforms(image)
            images.append(image)
        
        # label_0 indicates which image is preferred (0 = jpg_0, 1 = jpg_1)
        return {
            "preferred_images": images[0] if item['label_0'] == 0 else images[1],
            "less_preferred_images": images[1] if item['label_0'] == 0 else images[0],
            "prompt": item['caption']
        }


def collate_fn_dpo(examples):
    """Collate function for DPO training"""
    preferred_pixel_values = [example["preferred_images"] for example in examples]
    less_preferred_pixel_values = [example["less_preferred_images"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    
    preferred_pixel_values = torch.stack(preferred_pixel_values)
    preferred_pixel_values = preferred_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    less_preferred_pixel_values = torch.stack(less_preferred_pixel_values)
    less_preferred_pixel_values = less_preferred_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    batch = {
        "preferred_pixel_values": preferred_pixel_values,
        "less_preferred_pixel_values": less_preferred_pixel_values,
        "prompts": prompts
    }
    return batch


def compute_dpo_loss(
    model_preferred_pred,
    model_less_preferred_pred,
    target_preferred,
    target_less_preferred,
    ref_preferred_pred=None,
    ref_less_preferred_pred=None,
    beta=0.1,
    label_smoothing=0.0,
    reference_free=False
):
    """
    Compute DPO loss for diffusion models
    
    Args:
        model_preferred_pred: Model predictions for preferred images
        model_less_preferred_pred: Model predictions for less preferred images
        target_preferred: Ground truth targets for preferred images
        target_less_preferred: Ground truth targets for less preferred images
        ref_preferred_pred: Reference model predictions for preferred images (optional)
        ref_less_preferred_pred: Reference model predictions for less preferred images (optional)
        beta: DPO beta parameter (controls strength of KL penalty)
        label_smoothing: Label smoothing parameter
        reference_free: If True, use reference-free DPO variant
    """
    # Compute log probabilities as negative MSE with respect to TRUE TARGETS
    model_preferred_logprob = -F.mse_loss(model_preferred_pred, target_preferred, reduction='none').mean(dim=[1, 2, 3])
    model_less_preferred_logprob = -F.mse_loss(model_less_preferred_pred, target_less_preferred, reduction='none').mean(dim=[1, 2, 3])
    
    if reference_free:
        # Reference-free DPO: directly compare model preferences
        logits = beta * (model_preferred_logprob - model_less_preferred_logprob)
    else:
        # Standard DPO with reference model
        if ref_preferred_pred is None or ref_less_preferred_pred is None:
            raise ValueError("Reference predictions required when reference_free=False")
            
        # Reference model log probabilities also computed against TRUE TARGETS
        ref_preferred_logprob = -F.mse_loss(ref_preferred_pred, target_preferred, reduction='none').mean(dim=[1, 2, 3])
        ref_less_preferred_logprob = -F.mse_loss(ref_less_preferred_pred, target_less_preferred, reduction='none').mean(dim=[1, 2, 3])
        
        # DPO: compare log-ratio differences
        model_logratios = model_preferred_logprob - model_less_preferred_logprob
        ref_logratios = ref_preferred_logprob - ref_less_preferred_logprob
        logits = beta * (model_logratios - ref_logratios)
    
    # DPO loss with optional label smoothing
    if label_smoothing > 0:
        loss = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        loss = -F.logsigmoid(logits)
    
    # Return meaningful rewards (how well model performs on each type)
    chosen_rewards = beta * model_preferred_logprob.detach()
    rejected_rewards = beta * model_less_preferred_logprob.detach()
    
    preference_accuracy = (model_preferred_logprob > model_less_preferred_logprob).float().mean()
    logits_magnitude = torch.abs(logits).mean()
    
    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean(), preference_accuracy, logits_magnitude

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="FLUX DPO training script.")
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    
    # EMA arguments
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model for validation and final weights saving.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay factor (closer to 1 means slower updates).",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to DPO dataset created by create_dpo_dataset.py",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help="Whether to center crop the input images to the resolution",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    
    # DPO specific arguments
    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="DPO beta parameter (controls strength of KL penalty)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing parameter for DPO loss",
    )
    parser.add_argument(
        "--reference_free",
        action="store_true",
        help="Use reference-free DPO variant",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha parameter",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="The transformer modules to apply LoRA training on",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for FLUX",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help="Run validation every X epochs",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with the T5 text encoder",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Weighting scheme for flow matching",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean to use when using the 'logit_normal' weighting scheme",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std to use when using the 'logit_normal' weighting scheme",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
        )
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    # Make one log on every process with the configuration for debugging.
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
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id
    
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant
    )
    
    # Freeze base model weights
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # For mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move models to device and dtype
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Set up LoRA
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Make sure the trainable params are in float32
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
    elif args.mixed_precision == "bf16":
        cast_training_params([transformer], dtype=torch.float32)
    
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Dataset and DataLoader
    train_dataset = DPODataset(
        dataset_path=args.dataset_path,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_dpo,
        num_workers=args.dataloader_num_workers,
    )
    
    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Create EMA model for stable training
    if args.use_ema:
        ema_transformer = EMAModel(
            transformer,
            decay=args.ema_decay,
        )
    else:
        ema_transformer = None
    
    # Wrap ema_transformer with accelerator
    if args.use_ema:
        ema_transformer.to(accelerator.device)
    
    # Recalculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize trackers
    if accelerator.is_main_process:
        tracker_name = "flux-dpo-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running DPO training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Load checkpoint if resuming
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
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
        disable=not accelerator.is_local_main_process,
    )
    
    # Helper functions
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        if hasattr(text_encoder, "module"):
            dtype = text_encoder.module.dtype
        else:
            dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds


    def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

        if hasattr(text_encoder, "module"):
            dtype = text_encoder.module.dtype
        else:
            dtype = text_encoder.dtype
        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds


    def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if hasattr(text_encoders[0], "module"):
            dtype = text_encoders[0].module.dtype
        else:
            dtype = text_encoders[0].dtype

        pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoders[0],
            tokenizer=tokenizers[0],
            prompt=prompt,
            device=device if device is not None else text_encoders[0].device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
        )

        prompt_embeds = _encode_prompt_with_t5(
            text_encoder=text_encoders[1],
            tokenizer=tokenizers[1],
            max_sequence_length=max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else text_encoders[1].device,
            text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids
  
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # Save/load hooks for accelerator
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                
                weights.pop()
            
            # Save EMA state if enabled
            if args.use_ema:
                ema_state_dict = ema_transformer.state_dict()
                torch.save(ema_state_dict, os.path.join(output_dir, "ema_state_dict.pt"))
            
            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
    
    def load_model_hook(models, input_dir):
        transformer_ = None
        
        while len(models) > 0:
            model = models.pop()
            
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")
        
        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)
        
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        
        # Load EMA state if enabled
        if args.use_ema and os.path.exists(os.path.join(input_dir, "ema_state_dict.pt")):
            ema_state_dict = torch.load(os.path.join(input_dir, "ema_state_dict.pt"), map_location="cpu")
            ema_transformer.load_state_dict(ema_state_dict)
        
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Precompute VAE scale factor
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):


                # Extract batch data
                preferred_pixel_values = batch["preferred_pixel_values"].to(dtype=vae.dtype)
                less_preferred_pixel_values = batch["less_preferred_pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]
                
                # Encode prompts
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=[tokenizer_one, tokenizer_two],
                    prompt=prompts,
                    max_sequence_length=args.max_sequence_length,
                    device=accelerator.device
                )
                
                # Encode images to latents
                preferred_latents = vae.encode(preferred_pixel_values).latent_dist.sample()
                preferred_latents = (preferred_latents - vae_config_shift_factor) * vae_config_scaling_factor
                
                less_preferred_latents = vae.encode(less_preferred_pixel_values).latent_dist.sample()
                less_preferred_latents = (less_preferred_latents - vae_config_shift_factor) * vae_config_scaling_factor
                
                # Prepare latent image ids
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    preferred_latents.shape[0],
                    preferred_latents.shape[2] // 2,
                    preferred_latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                # Sample noise and timesteps
                noise = torch.randn_like(preferred_latents)
                bsz = preferred_latents.shape[0]
                
                # Sample timesteps
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=preferred_latents.device)
                
                # Add noise to latents
                sigmas = get_sigmas(timesteps, n_dim=preferred_latents.ndim, dtype=preferred_latents.dtype)
                noisy_preferred_latents = (1.0 - sigmas) * preferred_latents + sigmas * noise
                noisy_less_preferred_latents = (1.0 - sigmas) * less_preferred_latents + sigmas * noise
                
                # Pack latents
                packed_noisy_preferred = FluxPipeline._pack_latents(
                    noisy_preferred_latents,
                    batch_size=bsz,
                    num_channels_latents=noisy_preferred_latents.shape[1],
                    height=noisy_preferred_latents.shape[2],
                    width=noisy_preferred_latents.shape[3],
                )
                packed_noisy_less_preferred = FluxPipeline._pack_latents(
                    noisy_less_preferred_latents,
                    batch_size=bsz,
                    num_channels_latents=noisy_less_preferred_latents.shape[1],
                    height=noisy_less_preferred_latents.shape[2],
                    width=noisy_less_preferred_latents.shape[3],
                )
                
                # Handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(bsz)
                else:
                    guidance = None
                
                # Forward pass for preferred images
                model_pred_preferred = transformer(
                    hidden_states=packed_noisy_preferred,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                # Forward pass for less preferred images
                model_pred_less_preferred = transformer(
                    hidden_states=packed_noisy_less_preferred,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                # Unpack predictions
                model_pred_preferred = FluxPipeline._unpack_latents(
                    model_pred_preferred,
                    height=preferred_latents.shape[2] * vae_scale_factor,
                    width=preferred_latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                model_pred_less_preferred = FluxPipeline._unpack_latents(
                    model_pred_less_preferred,
                    height=less_preferred_latents.shape[2] * vae_scale_factor,
                    width=less_preferred_latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Compute target (flow matching)
                target_preferred = noise - preferred_latents
                target_less_preferred = noise - less_preferred_latents
                
                # Get reference model predictions if not reference-free
                if not args.reference_free:
                    with torch.no_grad():
                        # Temporarily disable LoRA for reference model prediction
                        transformer.disable_adapters()
                        
                        # Cast inputs to match the reference transformer's dtype
                        ref_hidden_states = packed_noisy_preferred.to(dtype=weight_dtype)
                        ref_timesteps = timesteps.to(dtype=weight_dtype)
                        ref_pooled_projections = pooled_prompt_embeds.to(dtype=weight_dtype)
                        ref_encoder_hidden_states = prompt_embeds.to(dtype=weight_dtype)
                        ref_txt_ids = text_ids.to(dtype=weight_dtype)
                        ref_img_ids = latent_image_ids.to(dtype=weight_dtype)
                        ref_guidance = guidance.to(dtype=weight_dtype) if guidance is not None else None
                        
                        ref_pred_preferred = transformer(
                            hidden_states=ref_hidden_states,
                            timestep=ref_timesteps / 1000,
                            guidance=ref_guidance,
                            pooled_projections=ref_pooled_projections,
                            encoder_hidden_states=ref_encoder_hidden_states,
                            txt_ids=ref_txt_ids,
                            img_ids=ref_img_ids,
                            return_dict=False,
                        )[0]
                        
                        # Do the same for less preferred inputs
                        ref_hidden_states = packed_noisy_less_preferred.to(dtype=weight_dtype)
                        
                        ref_pred_less_preferred = transformer(
                            hidden_states=ref_hidden_states,
                            timestep=ref_timesteps / 1000,
                            guidance=ref_guidance,
                            pooled_projections=ref_pooled_projections,
                            encoder_hidden_states=ref_encoder_hidden_states,
                            txt_ids=ref_txt_ids,
                            img_ids=ref_img_ids,
                            return_dict=False,
                        )[0]
                        
                        # Re-enable LoRA after reference prediction
                        transformer.enable_adapters()
                        
                        ref_pred_preferred = FluxPipeline._unpack_latents(
                            ref_pred_preferred,
                            height=preferred_latents.shape[2] * vae_scale_factor,
                            width=preferred_latents.shape[3] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )
                        ref_pred_less_preferred = FluxPipeline._unpack_latents(
                            ref_pred_less_preferred,
                            height=less_preferred_latents.shape[2] * vae_scale_factor,
                            width=less_preferred_latents.shape[3] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )
                else:
                    ref_pred_preferred = target_preferred
                    ref_pred_less_preferred = target_less_preferred
                
                # Compute DPO loss
                loss, chosen_rewards, rejected_rewards, pref_acc, logits_mag = compute_dpo_loss(
                    model_preferred_pred=model_pred_preferred.to(torch.float32),
                    model_less_preferred_pred=model_pred_less_preferred.to(torch.float32),
                    target_preferred=target_preferred.to(torch.float32),
                    target_less_preferred=target_less_preferred.to(torch.float32),
                    ref_preferred_pred=ref_pred_preferred.to(torch.float32) if not args.reference_free else None,
                    ref_less_preferred_pred=ref_pred_less_preferred.to(torch.float32) if not args.reference_free else None,
                    beta=args.beta,
                    label_smoothing=args.label_smoothing,
                    reference_free=args.reference_free
                )

                print(f"loss: {loss.item()}, chosen_rewards: {chosen_rewards.item()}, rejected_rewards: {rejected_rewards.item()}")
                print(f"Preferred MSE: {F.mse_loss(model_pred_preferred, target_preferred).item()}")
                print(f"Less preferred MSE: {F.mse_loss(model_pred_less_preferred, target_less_preferred).item()}")
                print(f"Preference margin: {(chosen_rewards - rejected_rewards).item()}")
                print(f"Preference accuracy: {pref_acc:.3f}, Logits magnitude: {logits_mag:.3f}")

                # Apply timestep weighting if specified
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = (loss * weighting).mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA model
                if args.use_ema:
                    ema_transformer.update()
        
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # Save checkpoint
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
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
            
            logs = {
                "loss": loss.detach().item(),
                "chosen_rewards": chosen_rewards.item(),
                "rejected_rewards": rejected_rewards.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
        
            # Validation
            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % 50 == 0:   # epoch % args.validation_epochs == 0:
                    with torch.no_grad():
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )

                        # Apply EMA weights for validation if enabled
                        if args.use_ema:
                            # Store current parameters before applying EMA weights
                            ema_transformer.apply_shadow()
                        

                        # Create pipeline for validation
                        pipeline = FluxPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=text_encoder_one,
                            text_encoder_2=text_encoder_two,
                            transformer=accelerator.unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                        
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        
                        # Generate validation images
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                        images = []
                        
                        for _ in range(args.num_validation_images):
                                image = pipeline(
                                    args.validation_prompt,
                                    num_inference_steps=30,
                                    generator=generator,
                                    guidance_scale=args.guidance_scale,
                                ).images[0]
                                images.append(image)
                        
                        # Restore original weights if using EMA
                        if args.use_ema:
                            ema_transformer.restore()
                        
                        # Log images
                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                np_images = np.stack([np.asarray(img) for img in images])
                                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                            if tracker.name == "wandb":
                                tracker.log(
                                    {
                                        "validation": [
                                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                            for i, image in enumerate(images)
                                        ]
                                    }
                                )

                        # Save images to disk
                        validation_dir = os.path.join(args.output_dir, "validation_images")
                        if accelerator.is_main_process:
                            os.makedirs(validation_dir, exist_ok=True)

                        for i, image in enumerate(images):
                            image_filename = f"{global_step:05d}_{i:02d}.jpg"
                            image_path = os.path.join(validation_dir, image_filename)
                            image.save(image_path)
                            logger.info(f"Saved validation image to {image_path}")
                        
                        del pipeline
                        torch.cuda.empty_cache()
    
    # Save the final LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Use EMA model if enabled for final weights
        if args.use_ema:
            # Apply EMA weights for final model
            ema_transformer.apply_shadow()
            
        transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        
        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        
        # Restore original weights if using EMA
        if args.use_ema:
            ema_transformer.restore()
        
        # Save model card
        if args.push_to_hub:
            save_model_card(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                images=None,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    
    accelerator.end_training()


def save_model_card(repo_id: str, images=None, base_model: str = None, repo_folder=None):
    """Save model card for HuggingFace Hub"""
    model_description = f"""
# FLUX DPO LoRA - {repo_id}

This is a FLUX model fine-tuned using Direct Preference Optimization (DPO).

## Model description

These are {repo_id} DPO LoRA weights for {base_model}.

The weights were trained using DPO to align the model with human preferences.

## Usage

```python
from diffusers import FluxPipeline
import torch

pipeline = FluxPipeline.from_pretrained(
    "{base_model}",
    torch_dtype=torch.bfloat16
)
pipeline.load_lora_weights("{repo_id}")
pipeline = pipeline.to("cuda")

image = pipeline(
    "your prompt here",
    guidance_scale=3.5,
    num_inference_steps=50,
).images[0]
```

## Training details

- Base model: {base_model}
- Training method: Direct Preference Optimization (DPO)
- LoRA rank: See training configuration

## License

Please adhere to the licensing terms of the base FLUX model.
"""
    
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
    )
    
    tags = [
        "text-to-image",
        "diffusers",
        "lora",
        "flux",
        "dpo",
        "preference-optimization",
        "template:sd-lora",
    ]
    
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


if __name__ == "__main__":
    args = parse_args()
    main(args)