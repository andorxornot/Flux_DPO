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


from diffusers import FluxPipeline

# --- Minimal hardcoded arguments ---
pretrained_model_id = "black-forest-labs/FLUX.1-dev"
lora_path = "models/my_first_flux_lora_v9_000008000.safetensors"
output_dir_fused = "models/fused_flux_full/"

# 1. Load full pipeline
pipe = FluxPipeline.from_pretrained(pretrained_model_id)

# 2. Load LoRA weights
print(f"Loading LoRA weights from {lora_path}")
pipe.load_lora_weights(lora_path)

# 3. Fuse LoRA layers with the base model
print("Fusing LoRA weights with base model")
pipe.fuse_lora()

# 4. Save the fused model
print(f"Saving fused model to {output_dir_fused}")
pipe.unload_lora_weights()
pipe.save_pretrained(output_dir_fused)

print("Model fusion completed successfully")