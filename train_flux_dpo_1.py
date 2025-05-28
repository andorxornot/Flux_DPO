#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
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

import argparse
import contextlib
import io
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
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParallelKwargs
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import PretrainedConfig, CLIPTokenizer, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import cast_training_params, free_memory

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = get_logger(__name__, log_level="INFO")

VALIDATION_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]


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


def log_validation(args, flux_transformer_model, vae_model, text_encoder_models, tokenizer_instances, accelerator_obj, weight_dtype_val, current_epoch, step=None, is_final_validation_run=False):
    logger.info(f"Running validation... \n Generating images with prompts:\n {VALIDATION_PROMPTS}.")

    text_encoder_one_val, text_encoder_two_val = text_encoder_models

    if is_final_validation_run:
        # For final validation, load LoRA into a fresh pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype_val,
            cache_dir=args.cache_dir,
        )
        pipeline.vae = vae_model
        pipeline.text_encoder = text_encoder_one_val
        pipeline.text_encoder_2 = text_encoder_two_val
        pipeline.load_lora_weights(args.output_dir)
    else:
        # During training, use the LoRA-adapted transformer (unwrapped)
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=accelerator_obj.unwrap_model(flux_transformer_model),
            vae=vae_model,
            text_encoder=text_encoder_one_val,
            text_encoder_2=text_encoder_two_val,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype_val,
            cache_dir=args.cache_dir,
        )

    pipeline = pipeline.to(accelerator_obj.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator_obj.device).manual_seed(args.seed) if args.seed else None
    images = []
    context = contextlib.nullcontext()

    guidance_scale = 0.0 if "dev" in args.pretrained_model_name_or_path.lower() else 5.0
    num_inference_steps = 30

    # Create validation directory if it doesn't exist
    validation_dir = os.path.join(args.output_dir, "validation_images")
    if accelerator_obj.is_main_process:
        os.makedirs(validation_dir, exist_ok=True)

    for i, prompt_text in enumerate(VALIDATION_PROMPTS):
        with context:
            prompt_embeds, pooled_embeds, _ = pipeline.encode_prompt(prompt_text, prompt_2=prompt_text)
            image = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            images.append(image)
            
            # Save the image
            if accelerator_obj.is_main_process and step is not None:
                image_filename = f"{step:05d}_{i:02d}.jpg"
                image_path = os.path.join(validation_dir, image_filename)
                image.save(image_path)
                logger.info(f"Saved validation image to {image_path}")

    tracker_key = "test" if is_final_validation_run else "validation"
    for tracker in accelerator_obj.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, current_epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}") for i, image in enumerate(images)
                    ]
                }
            )

    if is_final_validation_run:
        logger.info("Logging images without LoRA for comparison...")
        no_lora_pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae_model,
            text_encoder=text_encoder_one_val,
            text_encoder_2=text_encoder_two_val,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype_val,
            cache_dir=args.cache_dir,
        ).to(accelerator_obj.device)
        no_lora_pipeline.set_progress_bar_config(disable=True)

        no_lora_images = []
        for prompt_text in VALIDATION_PROMPTS:
            with context:
                prompt_embeds, pooled_embeds, _ = no_lora_pipeline.encode_prompt(prompt_text, prompt_2=prompt_text)
                image = no_lora_pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
                no_lora_images.append(image)
        del no_lora_pipeline

        for tracker in accelerator_obj.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in no_lora_images])
                tracker.writer.add_images("test_without_lora", np_images, current_epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "test_without_lora": [
                            wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                            for i, image in enumerate(no_lora_images)
                        ]
                    }
                )
    
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a DPO training script for FLUX (LoRA on Transformer only).")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained FLUX model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
            " Expected columns: 'jpg_0', 'jpg_1', 'label_0' (0 if jpg_0 is preferred), 'caption'."
        ),
    )
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="train",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=True,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dpo-lora-transformer-only",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader. DPO can be memory intensive."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=float,
        default=0.1,
        help="DPO KL Divergence penalty strength.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for LoRA.",
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
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help=("The alpha parameter for LoRA scaling (default: rank)."),
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers."
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated list of LoRA target modules for FluxTransformer2DModel. See FLUX DreamBooth script for examples.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for the T5 text encoder.",
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="flux-dpo-lora-transformer-only",
        help=("The name of the tracker to report results to."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    if args.lora_alpha is None:
        args.lora_alpha = args.rank

    # DPO-specific validations
    if args.beta_dpo <= 0:
        raise ValueError("beta_dpo must be positive")

    if args.train_batch_size < 1:
        raise ValueError("train_batch_size must be at least 1 for DPO")

    return args


def tokenize_captions_flux(tokenizer_list, examples, max_seq_len):
    captions = [caption for caption in examples["caption"]]
    tokens_one = tokenizer_list[0](
        captions, truncation=True, padding="max_length", max_length=tokenizer_list[0].model_max_length, return_tensors="pt"
    ).input_ids
    tokens_two = tokenizer_list[1](
        captions, truncation=True, padding="max_length", max_length=max_seq_len, return_tensors="pt"
    ).input_ids
    return tokens_one, tokens_two


@torch.no_grad()
def encode_prompt_flux(text_encoder_list, prompt_text_inputs_list, device_val, num_repeats_val=1):
    text_encoder_one, text_encoder_two = text_encoder_list
    clip_input_ids = prompt_text_inputs_list[0].to(device_val)
    pooled_prompt_embeds = text_encoder_one(clip_input_ids)[1]
    t5_input_ids = prompt_text_inputs_list[1].to(device_val)
    prompt_embeds = text_encoder_two(t5_input_ids)[0]

    if num_repeats_val > 1:
        prompt_embeds = prompt_embeds.repeat_interleave(num_repeats_val, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_repeats_val, dim=0)
    
    bsz, seq_len, _ = prompt_embeds.shape
    text_ids = torch.zeros(seq_len, 3, device=device_val, dtype=prompt_embeds.dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    # Enhanced dataset loading with better error handling
    try:
        if os.path.exists(args.dataset_name):
            logger.info(f"Loading dataset from local path: {args.dataset_name}")
            train_dataset_obj = load_from_disk(args.dataset_name)[args.dataset_split_name]
        else:
            logger.info(f"Loading dataset from HuggingFace Hub: {args.dataset_name}")
            train_dataset_obj = load_dataset(
                args.dataset_name,
                split=args.dataset_split_name,
                cache_dir=args.cache_dir,
            )
        
        # Validate dataset structure
        logger.info("Dataset loaded successfully. Validating structure...")
        logger.info(f"Dataset size: {len(train_dataset_obj)}")
        
        if hasattr(train_dataset_obj, 'column_names'):
            columns = train_dataset_obj.column_names
            logger.info(f"Dataset columns: {columns}")
            
            # Check for required columns
            required_columns = ['jpg_0', 'jpg_1', 'label_0', 'caption']
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {missing_columns}\n"
                    f"Available columns: {columns}\n"
                    "Please ensure your dataset contains: 'jpg_0', 'jpg_1', 'label_0', 'caption'"
                )
        else:
            raise ValueError("Dataset does not have a valid column structure")
            
        # Test first item
        first_item = train_dataset_obj[0]
        logger.info(f"First item keys: {list(first_item.keys())}")
        
        # Validate image data
        for key in ['jpg_0', 'jpg_1']:
            if not isinstance(first_item[key], bytes):
                raise ValueError(f"Column '{key}' should contain image bytes, got {type(first_item[key])}")
        
        # Validate label
        if first_item['label_0'] not in [0, 1]:
            raise ValueError(f"label_0 should be 0 or 1, got {first_item['label_0']}")
            
    except Exception as e:
        logger.error(f"Dataset error: {str(e)}")
        raise

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

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, cache_dir=args.cache_dir,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, cache_dir=args.cache_dir,
    )
    all_tokenizers = [tokenizer_one, tokenizer_two]

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
    )
    noise_scheduler_copy = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    all_text_encoders = [text_encoder_one, text_encoder_two]

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )

    # Freeze all parameters first
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move frozen models to device and cast to appropriate dtypes
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.to(accelerator.device, dtype=weight_dtype)

    if args.lora_target_modules is None:
        lora_target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
        ]
        logger.info(f"Using default LoRA target modules for FluxTransformer: {lora_target_modules}")
    else:
        lora_target_modules = [module.strip() for module in args.lora_target_modules.split(",")]

    # Fixed LoRA configuration - use rank for lora_alpha
    transformer_lora_config = LoraConfig(
        r=args.rank, 
        lora_alpha=args.rank,  # Fixed: use rank instead of args.lora_alpha
        init_lora_weights="gaussian",
        target_modules=lora_target_modules, 
        lora_dropout=args.lora_dropout,
    )
    flux_transformer.add_adapter(transformer_lora_config)

    # Fixed: Proper parameter casting for fp16 training
    if args.mixed_precision == "fp16":
        cast_training_params([flux_transformer], dtype=torch.float32)

    # Enhanced gradient checkpointing setup
    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        if hasattr(text_encoder_one, 'gradient_checkpointing_enable'):
            text_encoder_one.gradient_checkpointing_enable()
        if hasattr(text_encoder_two, 'gradient_checkpointing_enable'):
            text_encoder_two.gradient_checkpointing_enable()

    def unwrap_model_for_saving_hook(model_to_unwrap):
        model_to_unwrap = accelerator.unwrap_model(model_to_unwrap)
        model_to_unwrap = model_to_unwrap._orig_mod if is_compiled_module(model_to_unwrap) else model_to_unwrap
        return model_to_unwrap

    def save_model_hook(models_list, weights_list, output_dir_hook):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save_hook = None
            if len(models_list) == 1 and isinstance(models_list[0], type(unwrap_model_for_saving_hook(flux_transformer))):
                transformer_lora_layers_to_save_hook = get_peft_model_state_dict(models_list[0])
                weights_list.pop()
            else:
                for model_hook in models_list:
                    if isinstance(model_hook, type(unwrap_model_for_saving_hook(flux_transformer))):
                        transformer_lora_layers_to_save_hook = get_peft_model_state_dict(model_hook)
                        if weights_list: 
                            weights_list.pop(0)
                        break 
                if transformer_lora_layers_to_save_hook is None:
                    raise ValueError(f"Unexpected models in save_model_hook: {[m.__class__ for m in models_list]}")

            FluxPipeline.save_lora_weights(
                output_dir_hook,
                transformer_lora_layers=transformer_lora_layers_to_save_hook,
            )

    def load_model_hook(models_list, input_dir_hook):
        transformer_hook = None
        if len(models_list) == 1 and isinstance(models_list[0], type(unwrap_model_for_saving_hook(flux_transformer))):
            transformer_hook = models_list.pop()
        else:
            for i in range(len(models_list) -1, -1, -1):
                if isinstance(models_list[i], type(unwrap_model_for_saving_hook(flux_transformer))):
                    transformer_hook = models_list.pop(i)
                    logger.warning("load_model_hook: Multiple models found, attempting to load into FluxTransformer.")
                    break
            if transformer_hook is None:
                 raise ValueError(f"Unexpected models in load_model_hook: {[m.__class__ for m in models_list]}")

        lora_state_dict_hook = FluxPipeline.lora_state_dict(input_dir_hook)
        
        transformer_peft_state_dict = {}
        if "transformer" in lora_state_dict_hook:
            transformer_peft_state_dict = convert_unet_state_dict_to_peft(lora_state_dict_hook["transformer"])
        else:
            flat_transformer_dict = {k.replace("transformer.", ""): v for k,v in lora_state_dict_hook.items() if k.startswith("transformer.")}
            if flat_transformer_dict:
                 transformer_peft_state_dict = convert_unet_state_dict_to_peft(flat_transformer_dict)

        if transformer_hook is not None and transformer_peft_state_dict:
            from peft import set_peft_model_state_dict
            incompatible_keys = set_peft_model_state_dict(transformer_hook, transformer_peft_state_dict, adapter_name="default")
            if incompatible_keys and incompatible_keys.unexpected_keys:
                logger.warning(f"Unexpected keys when loading LoRA: {incompatible_keys.unexpected_keys}")
            # Ensure LoRA params are in correct dtype after loading
            if args.mixed_precision == "fp16":
                cast_training_params([transformer_hook], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class_adam = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class_adam = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install bitsandbytes.")

    params_to_optimize_val = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    if not params_to_optimize_val:
        raise ValueError("No LoRA parameters found to optimize. Check LoRA setup and target modules.")

    optimizer = optimizer_class_adam(
        params_to_optimize_val, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    train_resize_transform = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop_transform = transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution)
    train_flip_transform = transforms.RandomHorizontalFlip(p=1.0)
    to_tensor_transform = transforms.ToTensor()
    normalize_transform = transforms.Normalize([0.5], [0.5])

    def preprocess_train_fn(examples_dict):
        # Check for required keys
        required_keys = ["jpg_0", "jpg_1", "label_0", "caption"]
        missing_keys = [key for key in required_keys if key not in examples_dict]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in dataset: {missing_keys}.\n"
                f"Available keys: {list(examples_dict.keys())}\n"
                "The dataset must contain 'jpg_0', 'jpg_1' (image bytes), 'label_0' (0 if jpg_0 is preferred), and 'caption' columns."
            )

        images_w_bytes_list, images_l_bytes_list = [], []
        for i in range(len(examples_dict["jpg_0"])):
            if examples_dict["label_0"][i] == 0:
                images_w_bytes_list.append(examples_dict["jpg_0"][i])
                images_l_bytes_list.append(examples_dict["jpg_1"][i])
            else:
                images_w_bytes_list.append(examples_dict["jpg_1"][i])
                images_l_bytes_list.append(examples_dict["jpg_0"][i])

        images_w_pil = [Image.open(io.BytesIO(im_b)).convert("RGB") for im_b in images_w_bytes_list]
        images_l_pil = [Image.open(io.BytesIO(im_b)).convert("RGB") for im_b in images_l_bytes_list]
        
        processed_w_tensors, processed_l_tensors = [], []
        for img_w_item, img_l_item in zip(images_w_pil, images_l_pil):
            # Process img_w
            img_w_p = train_resize_transform(img_w_item)
            if not args.no_hflip and random.random() < 0.5: 
                img_w_p = train_flip_transform(img_w_p)
            img_w_p = train_crop_transform(img_w_p)
            processed_w_tensors.append(normalize_transform(to_tensor_transform(img_w_p)))
            # Process img_l
            img_l_p = train_resize_transform(img_l_item)
            if not args.no_hflip and random.random() < 0.5: 
                img_l_p = train_flip_transform(img_l_p)
            img_l_p = train_crop_transform(img_l_p)
            processed_l_tensors.append(normalize_transform(to_tensor_transform(img_l_p)))
        
        examples_dict["pixel_values_w"] = processed_w_tensors
        examples_dict["pixel_values_l"] = processed_l_tensors
        tokens_one_val, tokens_two_val = tokenize_captions_flux(all_tokenizers, examples_dict, args.max_sequence_length)
        examples_dict["input_ids_one"] = tokens_one_val
        examples_dict["input_ids_two"] = tokens_two_val
        return examples_dict

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset_obj = train_dataset_obj.shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset_obj = train_dataset_obj.with_transform(preprocess_train_fn)

    def collate_train_fn(examples_list_dicts):
        pixel_values_w_coll = torch.stack([ex["pixel_values_w"] for ex in examples_list_dicts])
        pixel_values_l_coll = torch.stack([ex["pixel_values_l"] for ex in examples_list_dicts])
        pixel_values_coll = torch.cat([pixel_values_w_coll, pixel_values_l_coll], dim=0)
        pixel_values_coll = pixel_values_coll.to(memory_format=torch.contiguous_format).float()
        input_ids_one_coll = torch.stack([ex["input_ids_one"] for ex in examples_list_dicts])
        input_ids_two_coll = torch.stack([ex["input_ids_two"] for ex in examples_list_dicts])
        return {
            "pixel_values": pixel_values_coll,
            "input_ids_one": input_ids_one_coll,
            "input_ids_two": input_ids_two_coll,
        }

    train_dataloader_obj = torch.utils.data.DataLoader(
        train_dataset_obj, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_train_fn, num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps_bool = False
    num_update_steps_per_epoch_val = math.ceil(len(train_dataloader_obj) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch_val
        overrode_max_train_steps_bool = True

    lr_scheduler_obj = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    flux_transformer, optimizer, train_dataloader_obj, lr_scheduler_obj = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader_obj, lr_scheduler_obj
    )

    num_update_steps_per_epoch_val = math.ceil(len(train_dataloader_obj) / args.gradient_accumulation_steps)
    if overrode_max_train_steps_bool:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch_val
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch_val)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    total_batch_size_val = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DPO training for FLUX (Transformer LoRA only) *****")
    logger.info(f"  Num preference pairs = {len(train_dataset_obj)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader_obj)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device (preference pairs) = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_val}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step_val = 0
    first_epoch_val = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path_val = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs_val = os.listdir(args.output_dir)
            dirs_val = [d for d in dirs_val if d.startswith("checkpoint")]
            dirs_val = sorted(dirs_val, key=lambda x: int(x.split("-")[1]))
            path_val = dirs_val[-1] if len(dirs_val) > 0 else None
        if path_val is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
            args.resume_from_checkpoint = None
            initial_global_step_val = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path_val}")
            accelerator.load_state(os.path.join(args.output_dir, path_val))
            global_step_val = int(path_val.split("-")[1])
            initial_global_step_val = global_step_val
            first_epoch_val = global_step_val // num_update_steps_per_epoch_val
    else:
        initial_global_step_val = 0

    progress_bar_obj = tqdm(
        range(0, args.max_train_steps), initial=initial_global_step_val, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    vae_shift = vae.config.shift_factor
    vae_scale = vae.config.scaling_factor

    def get_sigmas_fn(timesteps_in, n_dim_in=4, dtype_in=torch.float32, device_in=None):
        sigmas_all = noise_scheduler_copy.sigmas.to(device=device_in, dtype=dtype_in)
        schedule_all_t = noise_scheduler_copy.timesteps.to(device=device_in)
        timesteps_in = timesteps_in.to(device_in)
        step_indices_val = torch.empty_like(timesteps_in, dtype=torch.long)
        for i_val, t_val in enumerate(timesteps_in):
            step_indices_val[i_val] = (schedule_all_t == t_val).nonzero().item()
        sigma_out = sigmas_all[step_indices_val].flatten()
        while len(sigma_out.shape) < n_dim_in: 
            sigma_out = sigma_out.unsqueeze(-1)
        return sigma_out

    flux_transformer.train()
    for epoch_val in range(first_epoch_val, args.num_train_epochs):
        for step_val, batch_dict in enumerate(train_dataloader_obj):
            with accelerator.accumulate(flux_transformer):
                pixel_values_in = batch_dict["pixel_values"].to(dtype=vae.dtype)
                
                # Fixed VAE encoding with proper batch handling
                latents_list_val = []
                with torch.no_grad():
                    for i_val_vae in range(0, pixel_values_in.shape[0], args.vae_encode_batch_size):
                        end_idx = min(i_val_vae + args.vae_encode_batch_size, pixel_values_in.shape[0])
                        batch_pixels = pixel_values_in[i_val_vae:end_idx]
                        latents_list_val.append(
                            vae.encode(batch_pixels).latent_dist.sample()
                        )
                latents_in = torch.cat(latents_list_val, dim=0)
                latents_in = (latents_in - vae_shift) * vae_scale
                latents_in = latents_in.to(dtype=weight_dtype)

                bsz_pairs_val = batch_dict["input_ids_one"].shape[0]
                noise_single_val = torch.randn_like(latents_in[:bsz_pairs_val])
                noise_val = noise_single_val.repeat(2, 1, 1, 1)
                
                timesteps_indices_val = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz_pairs_val,), 
                    device=latents_in.device, dtype=torch.long
                )
                noise_scheduler_timesteps = noise_scheduler.timesteps.to(device=latents_in.device)
                actual_timesteps_single_val = noise_scheduler_timesteps[timesteps_indices_val]
                timesteps_val = actual_timesteps_single_val.repeat(2)

                sigmas_val = get_sigmas_fn(timesteps_val, n_dim_in=latents_in.ndim, 
                                         dtype_in=latents_in.dtype, device_in=latents_in.device)
                noisy_model_input_val = (1.0 - sigmas_val) * latents_in + sigmas_val * noise_val

                prompt_embeds_val, pooled_embeds_val, text_ids_val = encode_prompt_flux(
                    all_text_encoders, [batch_dict["input_ids_one"], batch_dict["input_ids_two"]],
                    accelerator.device, num_repeats_val=2
                )
                
                img_bsz_val, img_c_val, img_h_val, img_w_val = noisy_model_input_val.shape
                latent_img_ids_val = FluxPipeline._prepare_latent_image_ids(
                    img_bsz_val, img_h_val // 2, img_w_val // 2, accelerator.device, weight_dtype,
                )
                packed_noisy_input_val = FluxPipeline._pack_latents(
                    noisy_model_input_val, img_bsz_val, img_c_val, img_h_val, img_w_val
                )
                flow_target_val = noise_val - latents_in

                unconditional_guidance_scale = 0.0
                guidance_input = torch.full(
                    (timesteps_val.shape[0],),
                    unconditional_guidance_scale,
                    device=accelerator.device,
                    dtype=weight_dtype 
                )

                # Policy Model prediction
                policy_pred_packed_val = flux_transformer(
                    hidden_states=packed_noisy_input_val,
                    timestep=timesteps_val / 1000.0,
                    guidance=guidance_input, 
                    encoder_hidden_states=prompt_embeds_val,
                    pooled_projections=pooled_embeds_val,
                    txt_ids=text_ids_val,
                    img_ids=latent_img_ids_val,
                    return_dict=False,
                )[0]

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                policy_pred_unpacked_val = FluxPipeline._unpack_latents(
                    policy_pred_packed_val,
                    height=noisy_model_input_val.shape[2] * vae_scale_factor,
                    width=noisy_model_input_val.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                policy_mse_val = F.mse_loss(policy_pred_unpacked_val.float(), flow_target_val.float(), reduction="none")
                policy_mse_per_sample_val = policy_mse_val.mean(dim=list(range(1, len(policy_mse_val.shape))))
                policy_log_probs_w_val, policy_log_probs_l_val = (-policy_mse_per_sample_val).chunk(2)

                # Reference Model - Enhanced adapter management
                with torch.no_grad():
                    # Check if adapters are available and enabled
                    was_adapter_enabled = True
                    if hasattr(flux_transformer, 'peft_config'):
                        was_adapter_enabled = not flux_transformer.peft_config["default"].inference_mode
                    
                    flux_transformer.disable_adapters()
                    ref_pred_packed_val = flux_transformer(
                        hidden_states=packed_noisy_input_val,
                        timestep=timesteps_val / 1000.0,
                        guidance=guidance_input, 
                        encoder_hidden_states=prompt_embeds_val,
                        pooled_projections=pooled_embeds_val,
                        txt_ids=text_ids_val,
                        img_ids=latent_img_ids_val,
                        return_dict=False,
                    )[0]
                    
                    # Restore adapter state
                    if was_adapter_enabled:
                        flux_transformer.enable_adapters()
                    
                ref_pred_unpacked_val = FluxPipeline._unpack_latents(
                    ref_pred_packed_val,
                    height=noisy_model_input_val.shape[2] * vae_scale_factor,
                    width=noisy_model_input_val.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                ref_mse_val = F.mse_loss(ref_pred_unpacked_val.float(), flow_target_val.float(), reduction="none")
                ref_mse_per_sample_val = ref_mse_val.mean(dim=list(range(1, len(ref_mse_val.shape))))
                ref_log_probs_w_val, ref_log_probs_l_val = (-ref_mse_per_sample_val).chunk(2)
                
                pi_logratios_val = policy_log_probs_w_val - policy_log_probs_l_val
                ref_logratios_val = ref_log_probs_w_val - ref_log_probs_l_val
                logits_val = pi_logratios_val - ref_logratios_val
                loss_val = -F.logsigmoid(args.beta_dpo * logits_val).mean()

                raw_policy_mse_log = -0.5 * (policy_log_probs_w_val.mean() + policy_log_probs_l_val.mean())
                raw_ref_mse_log = -0.5 * (ref_log_probs_w_val.mean() + ref_log_probs_l_val.mean())
                implicit_acc_log = (logits_val > 0).float().mean()

                accelerator.backward(loss_val)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize_val, args.max_grad_norm)
                optimizer.step()
                lr_scheduler_obj.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar_obj.update(1)
                global_step_val += 1
                if accelerator.is_main_process:
                    if global_step_val % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            ckpts = os.listdir(args.output_dir)
                            ckpts = [d for d in ckpts if d.startswith("checkpoint")]
                            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
                            if len(ckpts) >= args.checkpoints_total_limit:
                                num_to_remove_ckpt = len(ckpts) - args.checkpoints_total_limit + 1
                                removing_ckpts = ckpts[0:num_to_remove_ckpt]
                                logger.info(f"Removing checkpoints: {', '.join(removing_ckpts)}")
                                for rm_ckpt in removing_ckpts: 
                                    shutil.rmtree(os.path.join(args.output_dir, rm_ckpt))
                        save_path_val = os.path.join(args.output_dir, f"checkpoint-{global_step_val}")
                        accelerator.save_state(save_path_val)
                        logger.info(f"Saved state to {save_path_val}")

                    if global_step_val % 10 == 0:  # Log every 10 steps
                        logger.info(f"Step {global_step_val} Debug Info:")
                        logger.info(f"  Policy W mean: {policy_log_probs_w_val.mean().item():.4f}")
                        logger.info(f"  Policy L mean: {policy_log_probs_l_val.mean().item():.4f}")
                        logger.info(f"  Ref W mean: {ref_log_probs_w_val.mean().item():.4f}")
                        logger.info(f"  Ref L mean: {ref_log_probs_l_val.mean().item():.4f}")
                        logger.info(f"  Pi logratios: {pi_logratios_val.mean().item():.4f}")
                        logger.info(f"  Ref logratios: {ref_logratios_val.mean().item():.4f}")
                        logger.info(f"  Final logits: {logits_val.mean().item():.4f}")
                        logger.info(f"  Logits > 0: {(logits_val > 0).sum().item()}/{len(logits_val)}")

                    if args.run_validation and global_step_val % args.validation_steps == 0:
                        logger.info(f"\nRunning validation at step {global_step_val}...")
                        log_validation(
                            args, flux_transformer, vae, all_text_encoders, all_tokenizers, 
                            accelerator, weight_dtype, epoch_val, step=global_step_val
                        )
                        # Make sure to return to training mode
                        flux_transformer.train()
                        logger.info("Validation completed. Resuming training...")

                logs_dict = {
                    "loss": loss_val.detach().item(), 
                    "policy_mse": raw_policy_mse_log.detach().item(),
                    "ref_mse": raw_ref_mse_log.detach().item(), 
                    "accuracy": implicit_acc_log.detach().item(),
                    "lr": lr_scheduler_obj.get_last_lr()[0],
                }
                progress_bar_obj.set_postfix(**logs_dict)
                accelerator.log(logs_dict, step=global_step_val)

                if global_step_val >= args.max_train_steps: 
                    break
        if global_step_val >= args.max_train_steps: 
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flux_transformer_final_save = accelerator.unwrap_model(flux_transformer)
        flux_transformer_final_save = flux_transformer_final_save.to(torch.float32) 
        
        transformer_lora_final_sd = get_peft_model_state_dict(flux_transformer_final_save)
        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_final_sd,
        )
        logger.info(f"Saved final LoRA weights for transformer to {args.output_dir}")

        if args.run_validation:
            log_validation(
                args, flux_transformer, vae, all_text_encoders, all_tokenizers,
                accelerator, weight_dtype, epoch_val, step=global_step_val, is_final_validation_run=True
            )

        if args.push_to_hub:
            logger.info("Pushing model to hub...")
            upload_folder(
                repo_id=repo_id, folder_path=args.output_dir,
                commit_message="End of DPO training for FLUX LoRA (transformer only)",
                ignore_patterns=["step_*", "epoch_*", "logs/*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)