import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from face_analysis_service import FaceAnalysisService

def create_similarity_report():
    # Initialize FaceAnalysisService
    service = FaceAnalysisService(silent=False)
    
    # Define the base path for validation images
    base_path = 'flux-dpo-lora-output/validation_images'
    
    # Create lists to store generation numbers and similarity scores
    generations = []
    scores_00 = []
    scores_01 = []
    scores_02 = []
    
    # Get all image files matching the pattern for _00.jpg
    pattern_00 = os.path.join(base_path, '*_00.jpg')
    image_files_00 = glob.glob(pattern_00)
    
    # Sort files by generation number
    image_files_00.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    
    print(f"Found {len(image_files_00)} images for analysis")
    
    for img_path_00 in image_files_00:
        # Extract generation number from filename
        base_name = os.path.basename(img_path_00)
        gen_num = int(base_name.split('_')[0])
        
        # Construct paths for the other image variants
        img_path_01 = img_path_00.replace('_00.jpg', '_01.jpg')
        img_path_02 = img_path_00.replace('_00.jpg', '_02.jpg')
        
        # Check if all required files exist
        if not all(os.path.exists(f) for f in [img_path_00, img_path_01, img_path_02]):
            print(f"Skipping generation {gen_num} - missing files")
            continue
        
        print(f"\nProcessing generation {gen_num}...")
        
        # Read images
        img_00 = cv2.imread(img_path_00)
        img_01 = cv2.imread(img_path_01)
        img_02 = cv2.imread(img_path_02)
        
        if img_00 is None or img_01 is None or img_02 is None:
            print(f"Error reading images for generation {gen_num}")
            continue
        
        # Analyze faces in all images
        result_00 = service.analyze_face_in_image(img_00, f"gen{gen_num}_00")
        result_01 = service.analyze_face_in_image(img_01, f"gen{gen_num}_01")
        result_02 = service.analyze_face_in_image(img_02, f"gen{gen_num}_02")
        
        # Check if analysis was successful for all images
        if not all(result['success'] for result in [result_00, result_01, result_02]):
            print(f"Face analysis failed for generation {gen_num}")
            if not result_00['success']:
                print(f"Image 00 error: {result_00.get('error', 'Unknown error')}")
            if not result_01['success']:
                print(f"Image 01 error: {result_01.get('error', 'Unknown error')}")
            if not result_02['success']:
                print(f"Image 02 error: {result_02.get('error', 'Unknown error')}")
            continue
        
        # Get similarity scores
        score_00 = result_00['similarity_score']
        score_01 = result_01['similarity_score']
        score_02 = result_02['similarity_score']
        
        print(f"Similarity scores for generation {gen_num}:")
        print(f"Image 00: {score_00:.4f}")
        print(f"Image 01: {score_01:.4f}")
        print(f"Image 02: {score_02:.4f}")
        
        # Store data for plotting
        generations.append(gen_num)
        scores_00.append(score_00)
        scores_01.append(score_01)
        scores_02.append(score_02)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(generations, scores_00, 'b-', label='Image 00')
    plt.plot(generations, scores_01, 'r-', label='Image 01')
    plt.plot(generations, scores_02, 'g-', label='Image 02')
    
    # Calculate and plot the average score line
    avg_scores = [(s0 + s1 + s2) / 3 for s0, s1, s2 in zip(scores_00, scores_01, scores_02)]
    plt.plot(generations, avg_scores, 'y-', linewidth=3, label='Average')
    
    plt.xlabel('Generation')
    plt.ylabel('Similarity Score')
    plt.title('Face Similarity Scores Across Generations')
    plt.legend()
    plt.grid(True)
    
    # Add markers to data points
    plt.plot(generations, scores_00, 'bo')
    plt.plot(generations, scores_01, 'ro')
    plt.plot(generations, scores_02, 'go')
    
    # Save the plot
    output_path = 'similarity_scores_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Calculate average scores
    avg_score_00 = np.mean(scores_00) if scores_00 else 0
    avg_score_01 = np.mean(scores_01) if scores_01 else 0
    avg_score_02 = np.mean(scores_02) if scores_02 else 0
    
    print("\nAverage similarity scores:")
    print(f"Image 00: {avg_score_00:.4f}")
    print(f"Image 01: {avg_score_01:.4f}")
    print(f"Image 02: {avg_score_02:.4f}")

if __name__ == "__main__":
    create_similarity_report() 