#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DPO (Direct Preference Optimization) Training Script for FLUX LoRA Models

This script implements DPO training for FLUX diffusion models using LoRA fine-tuning.
It supports preference-based training where the model learns to generate images
that are preferred over less preferred alternatives.

Key Features:
- DPO loss implementation for diffusion models
- FLUX transformer and text encoder LoRA training
- Consistent augmentation handling for image pairs
- Support for latent caching to improve training speed
- Validation and model card generation
"""

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import io
import torch.optim as optim

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from datasets import load_dataset, load_from_disk

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
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    validation_prompt=None,
    repo_folder=None,
):
    """Generate and save a model card for the trained LoRA weights."""
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append({
                "text": validation_prompt if validation_prompt else "A generated image",
                "output": {"url": f"image_{i}.png"}
            })

    model_description = f"""
# Flux DPO LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DPO LoRA weights for {base_model}.

The weights were trained using DPO-style preference tuning with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md), adapted for DPO.

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Trigger words

Use prompts as learned during DPO training. For example: `{validation_prompt if validation_prompt else "your prompt"}`

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the 🧨 diffusers library

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
).to('cuda')

pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else "your prompt"}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the documentation on loading LoRAs in diffusers

## License

Please adhere to the licensing terms as described here.
"""
    
    model_card_obj = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt if validation_prompt else "text-to-image",
        model_description=model_description,
        widget=widget_dict,
    )
    
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "dpo",
        "template:sd-lora",
    ]

    model_card_obj = populate_model_card(model_card_obj, tags=tags)
    model_card_obj.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two, args):
    """Load CLIP and T5 text encoders."""
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant
    )
    return text_encoder_one, text_encoder_two

def set_optimizer_lr(optimizer, new_lr):
    """
    Sets the learning rate for all parameter groups in the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        new_lr (float): The new learning rate to set.
    """
    if not isinstance(optimizer, optim.Optimizer):
        raise TypeError(f"optimizer should be an instance of torch.optim.Optimizer, got {type(optimizer)}")
    if not isinstance(new_lr, float) or new_lr < 0:
        raise ValueError(f"new_lr must be a non-negative float, got {new_lr}")

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(f"Optimizer learning rate changed to: {new_lr}")

def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    global_step=0,
    is_final_validation=False,
):
    """Run validation and log generated images."""
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            pipeline_args["prompt"], prompt_2=pipeline_args["prompt"]
        )
        
        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator
                ).images[0]
                images.append(image)

    # Save images to disk
    validation_dir = os.path.join(args.output_dir, "validation_images")
    if accelerator.is_main_process:
        os.makedirs(validation_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_filename = f"{global_step:05d}_{i:02d}.jpg"
        image_path = os.path.join(validation_dir, image_filename)
        image.save(image_path)
        logger.info(f"Saved validation image to {image_path}")

    # Log to trackers
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({
                phase_name: [
                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                    for i, image in enumerate(images)
                ]
            })

    del pipeline
    free_memory()
    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str,
    subfolder: str = "text_encoder"
):
    """Import the correct text encoder class based on model config."""
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


def parse_args(input_args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple example of a DPO training script for FLUX with LoRA."
    )
    
    # Model and dataset arguments
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
        help="Variant of the model files, e.g. 'fp16'",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help=(
            "The path to the DPO dataset (local path for load_from_disk or HF Hub ID for load_dataset). "
            "Dataset should contain 'jpg_0', 'jpg_1', 'caption', 'label_0' columns."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    
    # Text encoder arguments
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with the T5 text encoder. CLIP will use 77.",
    )
    
    # Validation arguments
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
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=10,
        help="Run validation every X epochs.",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers"
    )
    
    # DPO specific arguments
    parser.add_argument(
        "--dpo_beta",
        type=float,
        default=0.1,
        help="The beta hyperparameter for DPO loss.",
    )
    parser.add_argument(
        "--preference_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the DPO preference loss component.",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images, all images will be resized to this resolution"
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to center crop the input images to the resolution.",
    )
    parser.add_argument(
        "--allow_flipping",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to randomly flip images horizontally during training (applied consistently to pairs).",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10
    )
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
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
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
    
    # Optimizer arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for transformer LoRA.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Guidance scale for FLUX training if transformer.config.guidance_embeds is True.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use if train_text_encoder is True.",
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
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means data will be loaded in the main process.",
    )
    
    # Weighting scheme arguments
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="sigma_sqrt",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help='Weighting scheme for timesteps. "none" for uniform sampling and uniform loss.',
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the 'logit_normal' weighting scheme."
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the 'logit_normal' weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the 'mode' as the weighting_scheme.",
    )
    
    # Optimizer specific arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help='The optimizer type to use. Choose between ["AdamW", "prodigy"]',
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize. Ignored if optimizer is AdamW.",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay for Prodigy."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for transformer LoRA params."
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder LoRA params."
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help='The transformer modules to apply LoRA training on. Comma separated. E.g. - "to_k,to_q,to_v,to_out.0"',
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction for Prodigy. Ignored if optimizer is AdamW.",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate for Prodigy. Ignored if optimizer is AdamW.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    
    # Hub and logging arguments
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local output_dir.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents for preferred and less-preferred images.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are "tensorboard" (default), "wandb" and "comet_ml". Use "all" to report to all integrations.',
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help="Whether to upcast the trained transformer LoRA layers to float32 before saving.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_path is None:
        raise ValueError("You must specify --dataset_path for DPO training.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
        
    return args


class DPODataset(Dataset):
    """Dataset for DPO training with FLUX. Ensures consistent augmentations for pairs."""

    def __init__(
        self,
        dataset_path,
        size=1024,
        center_crop=True,
        allow_flipping=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.allow_flipping = allow_flipping

        # Load dataset
        if os.path.isdir(dataset_path):
            self.dataset = load_from_disk(dataset_path)
        else:
            # Assume it's a HuggingFace dataset
            self.dataset = load_dataset(dataset_path)

        # Use train split if dataset is a dict
        if isinstance(self.dataset, dict) and 'train' in self.dataset:
            self.dataset = self.dataset['train']
        elif isinstance(self.dataset, dict) and 'train' not in self.dataset:
            # If no 'train' split, try to use the first available split
            logger.warning("No 'train' split found in dataset. Using the first available split.")
            self.dataset = self.dataset[list(self.dataset.keys())[0]]

        # Setup transforms
        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        if center_crop:
            self.train_crop = transforms.CenterCrop(size)
        else:
            self.train_crop = transforms.RandomCrop(size)

        self.train_hflip = transforms.RandomHorizontalFlip(p=1.0)  # Always flips if called
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Determine shared augmentations for the pair
        apply_flip_to_pair = False
        if self.allow_flipping:
            apply_flip_to_pair = random.random() < 0.5

        crop_params_for_pair = None  # For RandomCrop

        # Process first image of the pair (from item['jpg_0'])
        img0_pil = Image.open(io.BytesIO(item['jpg_0']))
        img0_pil = exif_transpose(img0_pil)
        if not img0_pil.mode == "RGB":
            img0_pil = img0_pil.convert("RGB")
        img0_pil_resized = self.train_resize(img0_pil)

        if not self.center_crop:
            crop_params_for_pair = self.train_crop.get_params(img0_pil_resized, (self.size, self.size))

        # Apply augmentations to img0
        img0_aug = img0_pil_resized
        if apply_flip_to_pair:
            img0_aug = self.train_hflip(img0_aug)

        if self.center_crop:
            img0_aug = self.train_crop(img0_aug)
        else:  # RandomCrop
            y1, x1, h, w = crop_params_for_pair
            img0_aug = crop(img0_aug, y1, x1, h, w)
        img0_tensor = self.train_transforms(img0_aug)

        # Process second image of the pair (from item['jpg_1'])
        img1_pil = Image.open(io.BytesIO(item['jpg_1']))
        img1_pil = exif_transpose(img1_pil)
        if not img1_pil.mode == "RGB":
            img1_pil = img1_pil.convert("RGB")
        img1_pil_resized = self.train_resize(img1_pil)

        # Apply THE SAME augmentations to img1
        img1_aug = img1_pil_resized
        if apply_flip_to_pair:
            img1_aug = self.train_hflip(img1_aug)

        if self.center_crop:
            img1_aug = self.train_crop(img1_aug)
        else:  # RandomCrop
            y1, x1, h, w = crop_params_for_pair
            img1_aug = crop(img1_aug, y1, x1, h, w)
        img1_tensor = self.train_transforms(img1_aug)

        # label_0 indicates which image was originally preferred
        if item['label_0'] == 0:  # jpg_0 is preferred, jpg_1 is less_preferred
            final_preferred_img = img0_tensor
            final_less_preferred_img = img1_tensor
        else:  # jpg_1 is preferred, jpg_0 is less_preferred
            final_preferred_img = img1_tensor
            final_less_preferred_img = img0_tensor

        return {
            "preferred_images": final_preferred_img,
            "less_preferred_images": final_less_preferred_img,
            "prompt": item['caption']
        }


def collate_fn(examples):
    """Custom collate function for DPO training batches."""
    preferred_pixel_values = torch.stack([example["preferred_images"] for example in examples])
    less_preferred_pixel_values = torch.stack([example["less_preferred_images"] for example in examples])
    prompts = [example["prompt"] for example in examples]

    preferred_pixel_values = preferred_pixel_values.to(memory_format=torch.contiguous_format).float()
    less_preferred_pixel_values = less_preferred_pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "preferred_pixel_values": preferred_pixel_values,
        "less_preferred_pixel_values": less_preferred_pixel_values,
        "prompts": prompts,
    }
    return batch


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    """Tokenize a prompt with the given tokenizer."""
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
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    """Encode prompt using T5 text encoder."""
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

    dtype = text_encoder.module.dtype if hasattr(text_encoder, "module") else text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    if num_images_per_prompt > 1:
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
    """Encode prompt using CLIP text encoder."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,  # CLIP max length
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds_out = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    dtype = text_encoder.module.dtype if hasattr(text_encoder, "module") else text_encoder.dtype
    prompt_embeds = prompt_embeds_out.pooler_output  # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if num_images_per_prompt > 1:
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,  # List: [CLIP Encoder, T5 Encoder]
    tokenizers,  # List: [CLIP Tokenizer, T5 Tokenizer]
    prompt: str,  # Can be a list of prompts for batch
    max_sequence_length_t5,  # For T5
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,  # List: [CLIP ids, T5 ids]
):
    """Encode prompts using both CLIP and T5 text encoders."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].module.dtype if hasattr(text_encoders[0], "module") else text_encoders[0].dtype

    # CLIP Embeddings (Pooled)
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0] if tokenizers else None,
        prompt=prompt,
        device=device or text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    # T5 Embeddings (Sequence)
    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1] if tokenizers else None,
        max_sequence_length=max_sequence_length_t5,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device or text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    # FLUX specific text_ids
    text_ids_seq_len = prompt_embeds.shape[1]
    text_ids = torch.zeros(text_ids_seq_len, 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    """Main training function."""
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
        )
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("Mixed precision training with bfloat16 is not supported on MPS.")

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

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging.")

    # Setup logging
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

    if args.seed is not None:
        set_seed(args.seed)

    # Handle repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    # Import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )

    # Freeze base models
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("bf16 not supported on MPS.")

    # Move models to device with appropriate dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=torch.float32 if args.train_text_encoder else weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=torch.float32 if args.train_text_encoder else weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # Setup LoRA
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:  # Default FLUX LoRA target modules
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
        ]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules
    )
    transformer.add_adapter(transformer_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Saving and loading hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                elif args.train_text_encoder and isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None
        temp_models = list(models)
        
        for model_idx in range(len(temp_models)):
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif args.train_text_encoder and isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model

        lora_state_dict, _ = FluxPipeline.lora_state_dict(input_dir)

        if transformer_ is not None:
            transformer_state_dict = {
                f"{k.replace('transformer.', '')}": v 
                for k, v in lora_state_dict.items() 
                if k.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys and getattr(incompatible_keys, "unexpected_keys", None):
                logger.warning(f"Loading transformer adapter weights led to unexpected keys: {incompatible_keys.unexpected_keys}")

        if args.train_text_encoder and text_encoder_one_ is not None:
            text_encoder_one_state_dict = {
                f"{k.replace('text_encoder.', '')}": v 
                for k, v in lora_state_dict.items() 
                if k.startswith("text_encoder.")
            }
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Cast trainable LoRA params to float32 if mixed_precision is fp16
        if args.mixed_precision == "fp16":
            models_to_cast = []
            if transformer_ is not None: 
                models_to_cast.append(transformer_)
            if args.train_text_encoder and text_encoder_one_ is not None: 
                models_to_cast.append(text_encoder_one_)
            if models_to_cast:
                cast_training_params(models_to_cast, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Scale learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.train_text_encoder:
            args.text_encoder_lr = (
                args.text_encoder_lr * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

    # Ensure LoRA params are float32 for training if mixed precision is used
    if args.mixed_precision == "fp16":
        models_to_cast = [transformer]
        if args.train_text_encoder:
            models_to_cast.append(text_encoder_one)
        cast_training_params(models_to_cast, dtype=torch.float32)

    # Setup optimizer parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
        "weight_decay": args.adam_weight_decay
    }]

    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        params_to_optimize.append({
            "params": text_lora_parameters_one,
            "lr": args.text_encoder_lr,
            "weight_decay": args.adam_weight_decay_text_encoder,
        })

    # Create optimizer
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW.")
        args.optimizer = "adamw"
    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(f"use_8bit_adam ignored when optimizer is not AdamW. Optimizer: {args.optimizer}")

    if args.optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("Please install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`")
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
            optimizer_class = prodigyopt.Prodigy
        except ImportError:
            raise ImportError("Please install prodigyopt to use Prodigy: `pip install prodigyopt`")
        
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation
    train_dataset = DPODataset(
        dataset_path=args.dataset_path,
        size=args.resolution,
        center_crop=args.center_crop,
        allow_flipping=args.allow_flipping,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Cache latents if needed
    latents_preferred_cache = []
    latents_less_preferred_cache = []
    if args.cache_latents:
        logger.info("Caching latents...")
        vae.to(accelerator.device, dtype=weight_dtype)
        for batch_data in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                pref_pixels = batch_data["preferred_pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                less_pref_pixels = batch_data["less_preferred_pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                
                vae.eval()
                latents_preferred_cache.append(vae.encode(pref_pixels).latent_dist.sample())
                latents_less_preferred_cache.append(vae.encode(less_pref_pixels).latent_dist.sample())
        
        logger.info("Latents cached.")

    # Scheduler and training steps calculation
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with accelerator
    if args.train_text_encoder:
        transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
        text_encoder_one = accelerator.prepare_model(text_encoder_one, evaluation_mode=True)
        text_encoder_two = accelerator.prepare_model(text_encoder_two, evaluation_mode=True)

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = "dpo-flux-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Training info
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

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
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

    # VAE config for scaling latents
    vae_config_shift_factor = vae.config.shift_factor if vae is not None else 0.0
    vae_config_scaling_factor = vae.config.scaling_factor if vae is not None else 1.0
    vae_config_block_out_channels = vae.config.block_out_channels if vae is not None else [0]*4
    vae_latent_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1) if vae is not None else 8

    def get_sigmas_fn(timesteps, n_dim=4, dtype=torch.float32):
        """Get sigmas for given timesteps."""
        sigmas_val = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps_val = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps_val = timesteps.to(accelerator.device)
        step_indices = torch.tensor(
            [(schedule_timesteps_val == t).nonzero().item() for t in timesteps_val], 
            device=accelerator.device
        )
        
        sigma_val = sigmas_val[step_indices].flatten()
        while len(sigma_val.shape) < n_dim:
            sigma_val = sigma_val.unsqueeze(-1)
        return sigma_val

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.append(text_encoder_one)

            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]
                current_bs = len(prompts)

                # Encode prompts
                prepared_text_encoder_one = text_encoder_one
                prepared_text_encoder_two = text_encoder_two

                if args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[prepared_text_encoder_one, prepared_text_encoder_two],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        prompt=prompts,
                        max_sequence_length_t5=args.max_sequence_length,
                        device=accelerator.device,
                    )
                else:
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[prepared_text_encoder_one, prepared_text_encoder_two],
                            tokenizers=[tokenizer_one, tokenizer_two],
                            prompt=prompts,
                            max_sequence_length_t5=args.max_sequence_length,
                            device=accelerator.device,
                        )
                
                # Convert images to latent space
                if args.cache_latents and step < len(latents_preferred_cache):
                    latents_preferred = latents_preferred_cache[step].to(accelerator.device)
                    latents_less_preferred = latents_less_preferred_cache[step].to(accelerator.device)
                else:
                    pref_pixel_values = batch["preferred_pixel_values"].to(
                        device=accelerator.device, dtype=vae.dtype
                    )
                    less_pref_pixel_values = batch["less_preferred_pixel_values"].to(
                        device=accelerator.device, dtype=vae.dtype
                    )
                    if vae is None: 
                        raise ValueError("VAE is None, cannot encode pixels. Disable latent caching or ensure VAE is available.")
                    vae.eval()
                    latents_preferred = vae.encode(pref_pixel_values).latent_dist.sample()
                    latents_less_preferred = vae.encode(less_pref_pixel_values).latent_dist.sample()

                # Scale latents
                latents_preferred = (latents_preferred - vae_config_shift_factor) * vae_config_scaling_factor
                latents_less_preferred = (latents_less_preferred - vae_config_shift_factor) * vae_config_scaling_factor
                
                latents_preferred = latents_preferred.to(dtype=weight_dtype)
                latents_less_preferred = latents_less_preferred.to(dtype=weight_dtype)

                # Prepare latent image IDs
                latent_h, latent_w = latents_preferred.shape[2], latents_preferred.shape[3]
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    current_bs, latent_h // 2, latent_w // 2, accelerator.device, weight_dtype
                )

                # Sample noise and timesteps
                noise = torch.randn_like(latents_preferred)

                # Sample timesteps (shared for the pair)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=current_bs,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents_preferred.device)
                
                sigmas = get_sigmas_fn(timesteps, n_dim=latents_preferred.ndim, dtype=latents_preferred.dtype)

                # Add noise (Flow Matching)
                noisy_latents_preferred = (1.0 - sigmas) * latents_preferred + sigmas * noise
                noisy_latents_less_preferred = (1.0 - sigmas) * latents_less_preferred + sigmas * noise

                # Pack latents for FLUX
                packed_noisy_preferred = FluxPipeline._pack_latents(
                    noisy_latents_preferred, current_bs, noisy_latents_preferred.shape[1], latent_h, latent_w
                )
                packed_noisy_less_preferred = FluxPipeline._pack_latents(
                    noisy_latents_less_preferred, current_bs, noisy_latents_less_preferred.shape[1], latent_h, latent_w
                )

                # Guidance
                guidance_val = None
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance_val = torch.tensor([args.guidance_scale], device=accelerator.device).expand(current_bs)

                # Predict for preferred
                pred_preferred = transformer(
                    hidden_states=packed_noisy_preferred,
                    timestep=timesteps / 1000.0,
                    guidance=guidance_val,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                pred_preferred = FluxPipeline._unpack_latents(
                    pred_preferred, latent_h * vae_latent_scale_factor, latent_w * vae_latent_scale_factor, vae_latent_scale_factor
                )

                # Predict for less_preferred
                pred_less_preferred = transformer(
                    hidden_states=packed_noisy_less_preferred,
                    timestep=timesteps / 1000.0,
                    guidance=guidance_val,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                pred_less_preferred = FluxPipeline._unpack_latents(
                    pred_less_preferred, latent_h * vae_latent_scale_factor, latent_w * vae_latent_scale_factor, vae_latent_scale_factor
                )

                # Loss weighting
                loss_weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                #print("loss weighting",loss_weighting.float())

                # Flow matching targets
                target_preferred = noise - latents_preferred
                target_less_preferred = noise - latents_less_preferred
                
                '''
                                # MSE losses (per-sample, then mean)
                # Взвешенные MSE для основной задачи
                weighted_mse_preferred_raw = (loss_weighting.float() * (pred_preferred.float() - target_preferred.float()) ** 2).reshape(current_bs, -1).mean(dim=1)
                weighted_mse_less_preferred_raw = (loss_weighting.float() * (pred_less_preferred.float() - target_less_preferred.float()) ** 2).reshape(current_bs, -1).mean(dim=1)

                mse_preferred_for_anchor = weighted_mse_preferred_raw.mean()
                mse_less_preferred_for_anchor = weighted_mse_less_preferred_raw.mean()

                # НЕ взвешенные (или по-другому взвешенные) MSE для DPO-логитов
                # Вы можете либо полностью убрать loss_weighting, либо использовать loss_weighting=1.0
                unweighted_mse_preferred_raw = ((pred_preferred.float() - target_preferred.float()) ** 2).reshape(current_bs, -1).mean(dim=1)
                unweighted_mse_less_preferred_raw = ((pred_less_preferred.float() - target_less_preferred.float()) ** 2).reshape(current_bs, -1).mean(dim=1)

                # DPO loss
                # reward_chosen = -unweighted_mse_preferred_raw, reward_rejected = -unweighted_mse_less_preferred_raw
                # (reward_chosen - reward_rejected) = unweighted_mse_less_preferred_raw - unweighted_mse_preferred_raw
                preference_logits = args.dpo_beta * (unweighted_mse_less_preferred_raw - unweighted_mse_preferred_raw) # Используем невзвешенные MSE
                loss_dpo_preference = -torch.nn.functional.logsigmoid(preference_logits).mean()

                # Total loss
                total_loss = 0.5 * (mse_preferred_for_anchor + mse_less_preferred_for_anchor) + args.preference_loss_weight * loss_dpo_preference
                '''

                # MSE losses (per-sample, then mean)
                mse_preferred_raw = (loss_weighting.float() * (pred_preferred.float() - target_preferred.float()) ** 2).reshape(current_bs, -1).mean(dim=1)
                mse_less_preferred_raw = (loss_weighting.float() * (pred_less_preferred.float() - target_less_preferred.float()) ** 2).reshape(current_bs, -1).mean(dim=1)

                mse_preferred = mse_preferred_raw.mean()
                mse_less_preferred = mse_less_preferred_raw.mean()

                # DPO loss
                # We want to minimize mse_preferred and maximize mse_less_preferred
                # reward = -mse, so reward_chosen = -mse_preferred, reward_rejected = -mse_less_preferred
                # (reward_chosen - reward_rejected) = mse_less_preferred_raw - mse_preferred_raw
                preference_logits = args.dpo_beta * (mse_less_preferred_raw - mse_preferred_raw)
                loss_dpo_preference = -torch.nn.functional.logsigmoid(preference_logits).mean()

                # Total loss
                #total_loss = 0.5 * (mse_preferred + mse_less_preferred) + args.preference_loss_weight * loss_dpo_preference
                total_loss = mse_preferred + args.preference_loss_weight * loss_dpo_preference
                


                # more simple loss
                mse_preferred = (loss_weighting.float() * (pred_preferred.float() - target_preferred.float()) ** 2).mean()
                mse_less_preferred = (loss_weighting.float() * (pred_less_preferred.float() - target_less_preferred.float()) ** 2).mean()

                #total_loss = args.weight_for_preferred * mse_preferred +  args.weight_for_less_preferred * mse_less_preferred
                total_loss = 0.3 * mse_preferred + 0.7 * mse_less_preferred



                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    if args.train_text_encoder:
                        params_to_clip = itertools.chain(params_to_clip, text_encoder_one.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": total_loss.detach().item(),
                "mse_preferred": mse_preferred.detach().item(),
                "mse_less_preferred": mse_less_preferred.detach().item(),
                "dpo_preference_loss": loss_dpo_preference.detach().item(),
                "lr_transformer": optimizer.param_groups[0]['lr'],
            }
            if args.train_text_encoder and len(optimizer.param_groups) > 1:
                logs["lr_text_encoder"] = optimizer.param_groups[1]['lr']

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            print("logs",logs)
            

            if global_step >= args.max_train_steps:
                break
        
            # Validation at end of epoch
            if accelerator.is_main_process:
                if args.validation_prompt is not None and global_step % 50 == 0: #epoch % args.validation_epochs == 0:
                 
                    validation_pipeline = FluxPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        text_encoder=text_encoder_one,
                        text_encoder_2=text_encoder_two,
                        transformer=unwrap_model(transformer),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    validation_pipeline.to(accelerator.device)
                    
                    final_images = log_validation(
                        pipeline=validation_pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args={"prompt": args.validation_prompt},
                        epoch=epoch,
                        torch_dtype=weight_dtype,
                        global_step=global_step,
                        is_final_validation=False,
                    )

                    del validation_pipeline
                    free_memory()

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save final LoRA weights
        transformer_to_save = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer_to_save.to(torch.float32)
        
        transformer_lora_layers = get_peft_model_state_dict(transformer_to_save)
        text_encoder_lora_layers = None
        if args.train_text_encoder:
            text_encoder_one_to_save = unwrap_model(text_encoder_one)
            if args.upcast_before_saving:
                text_encoder_one_to_save.to(torch.float32)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one_to_save)

        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        # Final inference for model card
        if args.validation_prompt and args.num_validation_images > 0:
            final_pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype
            )
            final_pipeline.load_lora_weights(args.output_dir)
            final_pipeline.to(accelerator.device)
            
            final_images = log_validation(
                pipeline=final_pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args={"prompt": args.validation_prompt},
                epoch=args.num_train_epochs,
                torch_dtype=weight_dtype,
                global_step=global_step,
                is_final_validation=True,
            )
            del final_pipeline
            free_memory()
        else:
            final_images = None

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=final_images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of DPO training",
                ignore_patterns=["step_*", "epoch_*", "logs/**"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)