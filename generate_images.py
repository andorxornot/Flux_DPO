import torch
from diffusers import FluxPipeline
from typing import List, Optional
from PIL import Image
import os
import random

# Initialize pipeline
pipe = FluxPipeline.from_pretrained("/workspace/Flux_DPO/models/fused_flux_full", torch_dtype=torch.bfloat16)
pipe.to("cuda")

def generate(
    prompts: List[str],
    seed: Optional[int] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
) -> List[Image.Image]:
    """
    Generate images using FLUX pipeline  
     
    """
  
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    total_images = len(prompts)
    generated_images = []
    
   
    # Calculate current batch size
    current_batch_size = total_images
    current_prompts = prompts 
    
    # Generate batch of images
    outputs = pipe(
        prompt=current_prompts,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
    ).images
    
    generated_images.extend(outputs)
    
    return generated_images

# Example usage:
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Define base prompts that will be used with variations
    base_prompts = [
        "frontal a photo of ohwx man"
    ]
    
    # Generate 1000 prompts by adding variations to base prompts
    variations = ["in the style of film photography", "with cinematic lighting", 
                 "with dramatic shadows", "in high contrast", "with soft focus",
                 "with bokeh effect", "in muted colors", "with vibrant colors",
                 "in black and white", "with motion blur"]
    
    
    for i in range(220, 400):
        prompts = []
    
        variation = random.choice(variations)
        prompts.append(f"{base_prompts[0]}, {variation}")
        prompts.append(f"{base_prompts[0]}, {variation}")

        print(f"Generating {len(prompts)} images...")
        
        images = generate(
            prompts=prompts,
            #seed=42,
            height=1024,
            width=1024,
        )
        
        # Save generated images
        for idx, image in enumerate(images):
            image_filename = f"images/img_{i:02d}_{idx:02d}.jpg"
            prompt_filename = f"images/img_{i:02d}.txt"
            
            # Save the image
            image.save(image_filename, "JPEG", quality=95)
            
            # Save the prompt
            with open(prompt_filename, 'w') as f:
                f.write(prompts[idx])
                
            print(f"Generated {image_filename}")
         