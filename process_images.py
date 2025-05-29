import os
import cv2
from face_analysis_service import FaceAnalysisService
import glob
import json
import shutil

# Global variables for directories
IMAGES_DIR = 'images2'
DATASET_DIR = 'dataset2'

def process_images():
    # os make dirs
    os.makedirs(f'{DATASET_DIR}/preferred_images', exist_ok=True)
    os.makedirs(f'{DATASET_DIR}/less_preferred_images', exist_ok=True)

    # Initialize FaceAnalysisService
    service = FaceAnalysisService(silent=False)
    
    # Get list of all image pairs and text files
    image_files = glob.glob(f'{IMAGES_DIR}/img_*_00.jpg')
    
    # Initialize captions dictionary
    captions = {}
    
    for img0_path in image_files:
        # Extract N from the filename
        base_name = os.path.basename(img0_path)
        N = int(base_name.split('_')[1])
        
        # Construct paths for the second image and text file
        img1_path = f'{IMAGES_DIR}/img_{N:02d}_01.jpg'
        txt_path = f'{IMAGES_DIR}/img_{N:02d}.txt'
        
        # Check if all required files exist
        if not all(os.path.exists(f) for f in [img0_path, img1_path, txt_path]):
            print(f"Skipping {N} - missing files")
            continue
        
        print(f"\nProcessing image pair {N}...")
        
        # Read images
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        
        if img0 is None or img1 is None:
            print(f"Error reading images for {N}")
            continue
        
        # Read caption from text file
        try:
            with open(txt_path, 'r') as f:
                caption = f.read().strip()
        except Exception as e:
            print(f"Error reading caption for {N}: {e}")
            continue
            
        # Add caption to dictionary
        captions[f"image{N}"] = caption
        
        # Analyze faces in both images
        result0 = service.analyze_face_in_image(img0, f"{N}_0")
        result1 = service.analyze_face_in_image(img1, f"{N}_1")
        
        if not result0['success'] or not result1['success']:
            print(f"Face analysis failed for {N}")
            if not result0['success']:
                print(f"Image 0 error: {result0.get('error', 'Unknown error')}")
            if not result1['success']:
                print(f"Image 1 error: {result1.get('error', 'Unknown error')}")
            continue
        
        # Compare similarity scores
        score0 = result0['similarity_score']
        score1 = result1['similarity_score']
        
        print(f"Similarity scores:")
        print(f"Image 0: {score0:.4f}")
        print(f"Image 1: {score1:.4f}")
        
        # Determine which image is better
        better_id = 0 if score0 > score1 else 1
        print(f"Better image: {better_id} (score: {max(score0, score1):.4f})")
        
        # Save images to appropriate directories
        if better_id == 0:
            preferred_src = img0_path
            less_preferred_src = img1_path
        else:
            preferred_src = img1_path
            less_preferred_src = img0_path
            
  
        # Copy images to dataset directories with new names
        shutil.copy2(preferred_src, f'{DATASET_DIR}/preferred_images/image{N}.jpg')
        shutil.copy2(less_preferred_src, f'{DATASET_DIR}/less_preferred_images/image{N}.jpg')
    
    # Save captions to JSON file
    with open(f'{DATASET_DIR}/captions.json', 'w') as f:
        json.dump(captions, f, indent=4)
        
    print(f"\nProcessing complete. Dataset saved in '{DATASET_DIR}' directory.")

if __name__ == "__main__":
    process_images() 