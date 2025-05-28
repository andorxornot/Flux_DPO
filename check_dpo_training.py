#!/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import logging
from diffusers import FluxTransformer2DModel, FluxPipeline
from peft import LoraConfig
import traceback

# Configure logging to both file and console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("dpo_check.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

print("Starting DPO training check script...")

try:
    # Load the model and add LoRA adapter
    print("Loading model...")
    model_path = "/workspace/models/fused_flux_full"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        available_dirs = os.listdir("/workspace/models") if os.path.exists("/workspace/models") else []
        print(f"Available directories in /workspace/models: {available_dirs}")
        model_path = "/workspace/models/" + available_dirs[0] if available_dirs else model_path
        print(f"Trying with model path: {model_path}")
    
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer"
    )
    
    # Print model info
    print(f"Model type: {type(flux_transformer)}")
    print(f"Model device: {next(flux_transformer.parameters()).device}")
    
    # Freeze all parameters first
    flux_transformer.requires_grad_(False)
    print("All parameters frozen")
    
    # Add LoRA adapter
    print("Adding LoRA adapter...")
    lora_target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
    ]
    
    transformer_lora_config = LoraConfig(
        r=16,  # Using a smaller rank for quick testing
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
        lora_dropout=0.0,
    )
    
    flux_transformer.add_adapter(transformer_lora_config)
    print("LoRA adapter added")
    
    # Check if model has peft_config attribute
    if hasattr(flux_transformer, 'peft_config'):
        print("Model has peft_config attribute")
        print(f"PEFT config: {flux_transformer.peft_config}")
    else:
        print("Model does not have peft_config attribute!")
    
    # Check if LoRA parameters are trainable
    print("\nChecking LoRA parameters...")
    total_params = 0
    trainable_params = 0
    lora_param_count = 0
    
    for name, param in flux_transformer.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            lora_param_count += 1
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Number of LoRA parameter tensors: {lora_param_count}")
    
    # Create dummy inputs for testing
    print("\nCreating dummy inputs for testing...")
    batch_size = 2
    latent_channels = 4
    latent_height = 32
    latent_width = 32
    
    # Create dummy latents
    latents = torch.randn(batch_size, latent_channels, latent_height, latent_width)
    timesteps = torch.ones(batch_size) * 500
    
    # Create dummy text embeddings
    seq_len = 77
    text_embed_dim = 768
    prompt_embeds = torch.randn(batch_size, seq_len, text_embed_dim)
    pooled_embeds = torch.randn(batch_size, text_embed_dim)
    
    # Prepare latents for transformer input
    print("Preparing inputs for transformer...")
    packed_latents = FluxPipeline._pack_latents(
        latents, batch_size, latent_channels, latent_height, latent_width
    )
    
    # Create text and image IDs
    txt_ids = torch.zeros(seq_len, 3)
    img_ids = FluxPipeline._prepare_latent_image_ids(
        batch_size, latent_height // 2, latent_width // 2, "cpu", torch.float32
    )
    
    guidance = torch.zeros(batch_size)
    
    # Run with adapters enabled (policy model)
    print("\nRunning with adapters ENABLED (policy model)...")
    flux_transformer.train()
    flux_transformer.enable_adapters()
    
    with torch.no_grad():
        policy_outputs = flux_transformer(
            hidden_states=packed_latents,
            timestep=timesteps / 1000.0,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
    
    print(f"Policy output shape: {policy_outputs.shape}")
    policy_output_mean = policy_outputs.mean().item()
    policy_output_std = policy_outputs.std().item()
    print(f"Policy output stats: mean={policy_output_mean:.6f}, std={policy_output_std:.6f}")
    
    # Run with adapters disabled (reference model)
    print("\nRunning with adapters DISABLED (reference model)...")
    flux_transformer.disable_adapters()
    
    with torch.no_grad():
        reference_outputs = flux_transformer(
            hidden_states=packed_latents,
            timestep=timesteps / 1000.0,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
    
    print(f"Reference output shape: {reference_outputs.shape}")
    reference_output_mean = reference_outputs.mean().item()
    reference_output_std = reference_outputs.std().item()
    print(f"Reference output stats: mean={reference_output_mean:.6f}, std={reference_output_std:.6f}")
    
    # Compare outputs
    print("\nComparing policy and reference outputs...")
    output_diff = (policy_outputs - reference_outputs).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()
    
    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference: {mean_diff:.10f}")
    
    if max_diff < 1e-5:
        print("\n⚠️ ISSUE DETECTED: Policy and reference models produce nearly identical outputs!")
        print("This indicates that the LoRA adapters are not effectively modifying the model outputs.")
        print("Possible causes:")
        print("1. LoRA layers might not be properly attached to the model")
        print("2. LoRA weights might be too small to make a difference")
        print("3. The beta_dpo parameter might be too small (current: 0.01)")
        print("4. The learning rate might be too low (current: 1e-5)")
    else:
        print("\n✅ Policy and reference models produce different outputs.")
        print("This suggests the LoRA adapters are working correctly.")
    
    # Check if enable_adapters actually makes parameters trainable
    print("\nChecking adapter enable/disable functionality...")
    flux_transformer.disable_adapters()
    trainable_when_disabled = sum(p.requires_grad for _, p in flux_transformer.named_parameters())
    print(f"Trainable parameters when adapters disabled: {trainable_when_disabled}")
    
    flux_transformer.enable_adapters()
    trainable_when_enabled = sum(p.requires_grad for _, p in flux_transformer.named_parameters())
    print(f"Trainable parameters when adapters enabled: {trainable_when_enabled}")
    
    if trainable_when_enabled == trainable_when_disabled:
        print("\n⚠️ ISSUE DETECTED: enable_adapters() is not making parameters trainable!")
    else:
        print("\n✅ enable_adapters() correctly makes parameters trainable.")
    
    # Check beta_dpo value in the running script
    print("\nChecking beta_dpo value in the running process...")
    try:
        import subprocess
        result = subprocess.run(["ps", "aux", "|", "grep", "train_flux_dpo.py"], 
                               shell=True, capture_output=True, text=True)
        cmd_line = result.stdout
        print(f"Command line: {cmd_line}")
        
        if "--beta_dpo 0.01" in cmd_line:
            print("\n⚠️ ISSUE DETECTED: beta_dpo is set to 0.01, which is very small!")
            print("Recommended: Increase beta_dpo to at least 0.1 or higher (e.g., 0.5 or 1.0)")
    except Exception as e:
        print(f"Error checking beta_dpo: {str(e)}")
    
    print("\nDone!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print(traceback.format_exc()) 