import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from face_analysis_service import FaceAnalysisService

def create_similarity_report(image_variants):
    # Initialize FaceAnalysisService
    service = FaceAnalysisService(silent=False)
    
    # Define the base path for validation images
    base_path = '/workspace/Flux_DPO/flux-dpo-lora-output/validation_images'
    
    # Create lists to store generation numbers and similarity scores
    generations = []
    scores = [[] for _ in range(image_variants)]
    
    # Get all image files matching the pattern for _00.jpg (first variant)
    pattern_base = os.path.join(base_path, '*_00.jpg')
    image_files_base = glob.glob(pattern_base)
    
    # Sort files by generation number
    image_files_base.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    
    print(f"Found {len(image_files_base)} images for analysis")
    
    for img_path_base in image_files_base:
        # Extract generation number from filename
        base_name = os.path.basename(img_path_base)
        gen_num = int(base_name.split('_')[0])
        
        # Construct paths for all image variants
        img_paths = []
        for i in range(image_variants):
            variant_path = img_path_base.replace('_00.jpg', f'_{i:02d}.jpg')
            img_paths.append(variant_path)
        
        # Check if all required files exist
        if not all(os.path.exists(f) for f in img_paths):
            print(f"Skipping generation {gen_num} - missing files")
            continue
        
        print(f"\nProcessing generation {gen_num}...")
        
        # Read images
        images = []
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Error reading image {path}")
                break
            images.append(img)
        
        if len(images) != image_variants:
            print(f"Error reading images for generation {gen_num}")
            continue
        
        # Analyze faces in all images
        results = []
        for i, img in enumerate(images):
            result = service.analyze_face_in_image(img, f"gen{gen_num}_{i:02d}")
            results.append(result)
        
        # Check if analysis was successful for all images
        if not all(result['success'] for result in results):
            print(f"Face analysis failed for generation {gen_num}")
            for i, result in enumerate(results):
                if not result['success']:
                    print(f"Image {i:02d} error: {result.get('error', 'Unknown error')}")
            continue
        
        # Get similarity scores
        similarity_scores = [result['similarity_score'] for result in results]
        
        print(f"Similarity scores for generation {gen_num}:")
        for i, score in enumerate(similarity_scores):
            print(f"Image {i:02d}: {score:.4f}")
        
        # Store data for plotting
        generations.append(gen_num)
        for i, score in enumerate(similarity_scores):
            scores[i].append(score)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each variant line with interpolated colors between cyan (0,255,255) and blue (0,0,255)
    start_color = np.array([0, 255, 255])  # cyan in RGB
    end_color = np.array([0, 0, 255])      # blue in RGB
    
    for i in range(image_variants):
        # Interpolate between start and end color
        t = i / max(1, image_variants - 1)  # normalize to [0, 1]
        rgb_color = start_color * (1 - t) + end_color * t
        # Convert to matplotlib color format (0-1 range)
        color = rgb_color / 255.0
        
        plt.plot(generations, scores[i], '-', color=color, label=f'Image {i:02d}')
        plt.plot(generations, scores[i], 'o', color=color)  # Add markers
    
    # Calculate and plot the average score line
    if generations:
        avg_scores = []
        for i in range(len(generations)):
            variant_scores = [scores[j][i] for j in range(image_variants)]
            avg_scores.append(sum(variant_scores) / len(variant_scores))
        plt.plot(generations, avg_scores, 'y-', linewidth=3, label='Average')
    
    plt.xlabel('Generation')
    plt.ylabel('Similarity Score')
    plt.title('Face Similarity Scores Across Generations')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_path = 'similarity_scores_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Calculate average scores
    print("\nAverage similarity scores:")
    for i in range(image_variants):
        avg_score = np.mean(scores[i]) if scores[i] else 0
        print(f"Image {i:02d}: {avg_score:.4f}")

if __name__ == "__main__":
    create_similarity_report(6)  # Default to 3 variants (00, 01, 02) 