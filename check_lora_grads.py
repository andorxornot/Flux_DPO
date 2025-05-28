#!/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import logging
from diffusers import FluxTransformer2DModel
from peft import LoraConfig
import traceback

# Configure logging to both file and console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("lora_check.log"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

print("Starting LoRA gradient check script...")

try:
    # Load the model and add LoRA adapter
    print("Loading model...")
    logger.info("Loading model...")
    model_path = "/workspace/models/fused_flux_full"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        logger.error(f"Model path {model_path} does not exist!")
        available_dirs = os.listdir("/workspace/models") if os.path.exists("/workspace/models") else []
        print(f"Available directories in /workspace/models: {available_dirs}")
        logger.info(f"Available directories in /workspace/models: {available_dirs}")
        model_path = "/workspace/models/" + available_dirs[0] if available_dirs else model_path
        print(f"Trying with model path: {model_path}")
        logger.info(f"Trying with model path: {model_path}")
    
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer"
    )
    
    # Print model info
    print(f"Model type: {type(flux_transformer)}")
    logger.info(f"Model type: {type(flux_transformer)}")
    print(f"Model device: {next(flux_transformer.parameters()).device}")
    logger.info(f"Model device: {next(flux_transformer.parameters()).device}")
    
    # Freeze all parameters first
    flux_transformer.requires_grad_(False)
    print("All parameters frozen")
    logger.info("All parameters frozen")
    
    # Add LoRA adapter
    print("Adding LoRA adapter...")
    logger.info("Adding LoRA adapter...")
    lora_target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
    ]
    
    # Check if target modules exist in the model
    found_modules = []
    for name, _ in flux_transformer.named_modules():
        for target in lora_target_modules:
            if target in name:
                found_modules.append(name)
                break
    
    print(f"Found {len(found_modules)} modules matching target patterns")
    logger.info(f"Found {len(found_modules)} modules matching target patterns")
    print(f"Example found modules: {found_modules[:5]}")
    logger.info(f"Example found modules: {found_modules[:5]}")
    
    transformer_lora_config = LoraConfig(
        r=16,  # Using a smaller rank for quick testing
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
        lora_dropout=0.0,
    )
    
    flux_transformer.add_adapter(transformer_lora_config)
    print("LoRA adapter added")
    logger.info("LoRA adapter added")
    
    # Check if model has peft_config attribute
    if hasattr(flux_transformer, 'peft_config'):
        print("Model has peft_config attribute")
        logger.info("Model has peft_config attribute")
        print(f"PEFT config: {flux_transformer.peft_config}")
        logger.info(f"PEFT config: {flux_transformer.peft_config}")
    else:
        print("Model does not have peft_config attribute!")
        logger.warning("Model does not have peft_config attribute!")
    
    # Check if LoRA parameters are trainable
    print("\nChecking LoRA parameters...")
    logger.info("\nChecking LoRA parameters...")
    total_params = 0
    trainable_params = 0
    lora_param_count = 0
    lora_param_names = []
    
    for name, param in flux_transformer.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            lora_param_count += 1
            lora_param_names.append(name)
    
    print(f"Total parameters: {total_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Number of LoRA parameter tensors: {lora_param_count}")
    logger.info(f"Number of LoRA parameter tensors: {lora_param_count}")
    
    # Print some example LoRA parameter names
    if lora_param_names:
        print("\nExample LoRA parameter names:")
        logger.info("\nExample LoRA parameter names:")
        for name in lora_param_names[:10]:  # Print first 10
            print(f"  - {name}")
            logger.info(f"  - {name}")
        if len(lora_param_names) > 10:
            print(f"  ... and {len(lora_param_names) - 10} more")
            logger.info(f"  ... and {len(lora_param_names) - 10} more")
    else:
        print("No trainable LoRA parameters found!")
        logger.error("No trainable LoRA parameters found!")
    
    # Test if gradients flow through LoRA parameters
    print("\nTesting gradient flow...")
    logger.info("\nTesting gradient flow...")
    # Create a dummy input
    batch_size = 2
    hidden_states = torch.randn(batch_size, 4, 32, 32)  # Adjust dimensions as needed
    timestep = torch.ones(batch_size) * 0.5
    guidance = torch.zeros(batch_size)
    encoder_hidden_states = torch.randn(batch_size, 77, 768)  # Adjust dimensions as needed
    pooled_projections = torch.randn(batch_size, 768)  # Adjust dimensions as needed
    txt_ids = torch.zeros(77, 3)
    img_ids = torch.zeros(32, 32)
    
    # Forward pass
    flux_transformer.train()
    try:
        outputs = flux_transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        
        # Compute loss and backward
        loss = outputs.mean()
        loss.backward()
        
        # Check if gradients are computed
        grad_exists = 0
        grad_zero = 0
        params_without_grad = []
        
        for name, param in flux_transformer.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    if param.grad.abs().sum().item() > 0:
                        grad_exists += 1
                    else:
                        grad_zero += 1
                        print(f"Parameter {name} has zero gradient")
                        logger.warning(f"Parameter {name} has zero gradient")
                else:
                    params_without_grad.append(name)
                    print(f"Parameter {name} has no gradient")
                    logger.warning(f"Parameter {name} has no gradient")
        
        if params_without_grad:
            print(f"{len(params_without_grad)} parameters have no gradients")
            logger.warning(f"{len(params_without_grad)} parameters have no gradients")
            print(f"Example parameters without gradients: {params_without_grad[:5]}")
            logger.warning(f"Example parameters without gradients: {params_without_grad[:5]}")
        
        print(f"\nParameters with non-zero gradients: {grad_exists}/{lora_param_count}")
        logger.info(f"\nParameters with non-zero gradients: {grad_exists}/{lora_param_count}")
        print(f"Parameters with zero gradients: {grad_zero}/{lora_param_count}")
        logger.info(f"Parameters with zero gradients: {grad_zero}/{lora_param_count}")
        
        # Check if adapters can be disabled/enabled
        print("\nTesting adapter disable/enable...")
        logger.info("\nTesting adapter disable/enable...")
        flux_transformer.disable_adapters()
        print("Adapters disabled")
        logger.info("Adapters disabled")
        
        # Check which parameters are trainable after disabling
        trainable_after_disable = sum(p.requires_grad for _, p in flux_transformer.named_parameters())
        print(f"Trainable parameters after disable: {trainable_after_disable}")
        logger.info(f"Trainable parameters after disable: {trainable_after_disable}")
        
        flux_transformer.enable_adapters()
        print("Adapters enabled")
        logger.info("Adapters enabled")
        
        # Check which parameters are trainable after enabling
        trainable_after_enable = sum(p.requires_grad for _, p in flux_transformer.named_parameters())
        print(f"Trainable parameters after enable: {trainable_after_enable}")
        logger.info(f"Trainable parameters after enable: {trainable_after_enable}")
        
        if trainable_after_enable != lora_param_count:
            print(f"Mismatch in trainable parameters after re-enabling: {trainable_after_enable} vs {lora_param_count}")
            logger.error(f"Mismatch in trainable parameters after re-enabling: {trainable_after_enable} vs {lora_param_count}")
        
    except Exception as e:
        print(f"Error during forward/backward pass: {str(e)}")
        logger.error(f"Error during forward/backward pass: {str(e)}")
        print(traceback.format_exc())
        logger.error(traceback.format_exc())
    
    print("\nDone!")
    logger.info("\nDone!")
except Exception as e:
    print(f"Error: {str(e)}")
    logger.error(f"Error: {str(e)}")
    print(traceback.format_exc())
    logger.error(traceback.format_exc()) 