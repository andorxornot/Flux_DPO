import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from typing import List, Optional
from PIL import Image
import os
import random

# Global constants
IMAGE_DIR = "images2"

# Initialize pipeline
pipe = FluxPipeline.from_pretrained("/workspace/Flux_DPO/models/fused_flux_full", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Initialize img2img pipeline from the already loaded pipe
#img2img_pipe = FluxImg2ImgPipeline.from_pipe(pipe)
img2img_pipe = FluxImg2ImgPipeline.from_pretrained("/workspace/Flux_DPO/models/fused_flux_full", torch_dtype=torch.bfloat16)
img2img_pipe.to("cuda")

def generate(
    prompts: List[str],
    seed: Optional[int] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
) -> List[Image.Image]:
    """
    Generate images using FLUX pipeline with seed switching
    First generates an image with text-to-image, then uses img2img for the second image
    """
  
    # Set seed if provided
    if seed is not None:
        initial_seed = seed
    else:
        initial_seed = random.randint(0, 2**32 - 1)

    generated_images = []
 
    # Set initial seed and get generator
    generator = torch.Generator("cuda").manual_seed(initial_seed)

    # Generate first image with callback
    outputs = pipe(
        prompt=prompts[0],
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator
    ).images
    generated_images.extend(outputs)

    second_seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(second_seed)
  
    # Use img2img with the first generated image as input
    outputs = img2img_pipe(
        prompt=prompts[1],
        image=outputs[0],  # Use the first generated image as input
        strength=0.85,  # Control how much to transform the image (0.5 = 50% transformation)
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator
    ).images
    generated_images.extend(outputs)
    
    return generated_images

# Example usage:
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Define base prompts that will be used with variations
    base_prompts = [
        "frontal a photo of ohwx man"
    ]
    
    # Generate 1000 prompts by adding variations to base prompts
    variations = ["in the style of film photography", "with cinematic lighting", 
                 "with dramatic shadows", "in high contrast", "with soft focus",
                 "with bokeh effect", "in muted colors", "with vibrant colors",
                 "in black and white", "with motion blur"]
    
    
    for i in range(0, 400):
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
            image_filename = f"{IMAGE_DIR}/img_{i:02d}_{idx:02d}.jpg"
            prompt_filename = f"{IMAGE_DIR}/img_{i:02d}.txt"
            
            # Save the image
            image.save(image_filename, "JPEG", quality=95)
            
            # Save the prompt
            with open(prompt_filename, 'w') as f:
                f.write(prompts[idx])
                
            print(f"Generated {image_filename}")
         