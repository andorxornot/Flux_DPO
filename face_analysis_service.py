import cv2
import numpy as np
from insightface import app as face_app
from PIL import Image
import os
import base64
from io import BytesIO
from typing import List, Tuple, Optional, Dict
import json
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys


class FaceAnalysisService:
    def __init__(self, model_root='./models', silent=False):
        """Initialize the face analysis service with ArcFace model"""
        self.silent = silent
        
        # Ensure models directory exists
        os.makedirs(model_root, exist_ok=True)
        os.makedirs('crop', exist_ok=True)  # For testing face crops
        
        # Cache file for reference embeddings
        self.reference_cache_file = 'reference_embeddings.pkl'
        
        # Initialize InsightFace app (suppress output if silent)
        if self.silent:
            # Redirect stdout to suppress InsightFace initialization messages
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        try:
            self.app = face_app.FaceAnalysis(providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        finally:
            if self.silent:
                sys.stdout.close()
                sys.stdout = original_stdout
        
        # Cache for reference face embeddings - now storing all embeddings
        self.reference_embeddings = []
        self.reference_faces_processed = False
        
        # Load reference faces (from cache or process fresh)
        self._load_reference_faces()
    
    def _load_reference_faces(self):
        """Load reference faces from cache or process them fresh"""
        reference_dir = 'reference_face'
        
        # Check if cache exists and is newer than reference directory
        if self._should_use_cache(reference_dir):
            if self._load_from_cache():
                if not self.silent:
                    print(f"✅ Loaded reference embeddings from cache")
                    print(f"Number of reference embeddings: {len(self.reference_embeddings)}")
                return
        
        # Process reference faces fresh
        self._process_reference_faces(reference_dir)
    
    def _should_use_cache(self, reference_dir):
        """Check if we should use cached embeddings"""
        if not os.path.exists(self.reference_cache_file):
            return False
        
        if not os.path.exists(reference_dir):
            return False
        
        # Check if cache is newer than all reference images
        cache_time = os.path.getmtime(self.reference_cache_file)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for filename in os.listdir(reference_dir):
            if filename.lower().endswith(image_extensions):
                filepath = os.path.join(reference_dir, filename)
                if os.path.getmtime(filepath) > cache_time:
                    if not self.silent:
                        print(f"Reference image {filename} is newer than cache, reprocessing...")
                    return False
        
        return True
    
    def _load_from_cache(self):
        """Load reference embeddings from cache file"""
        try:
            with open(self.reference_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.reference_embeddings = cache_data['embeddings']
                self.reference_faces_processed = True
                return True
        except Exception as e:
            if not self.silent:
                print(f"Failed to load cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Save reference embeddings to cache file"""
        try:
            cache_data = {
                'embeddings': self.reference_embeddings,
                'processed_count': len([f for f in os.listdir('crop') if f.startswith('reference_')])
            }
            with open(self.reference_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            if not self.silent:
                print(f"💾 Saved reference embeddings to cache")
        except Exception as e:
            if not self.silent:
                print(f"Failed to save cache: {e}")
    
    def _process_reference_faces(self, reference_dir):
        """Process reference faces from reference_face/ directory"""
        if not os.path.exists(reference_dir):
            if not self.silent:
                print(f"Warning: Reference face directory '{reference_dir}' not found")
            return
        
        self.reference_embeddings = []
        processed_count = 0
        
        # Get all image files in reference directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(reference_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            if not self.silent:
                print(f"Warning: No image files found in '{reference_dir}'")
            return
        
        if not self.silent:
            print(f"Processing {len(image_files)} reference faces...")
        
        for filename in image_files:
            filepath = os.path.join(reference_dir, filename)
            try:
                # Load image
                img = cv2.imread(filepath)
                if img is None:
                    if not self.silent:
                        print(f"Warning: Could not load {filename}")
                    continue
                
                # Detect faces
                faces = self.app.get(img)
                if len(faces) == 0:
                    if not self.silent:
                        print(f"Warning: No face detected in {filename}")
                    continue
                
                # Get the largest face (by bounding box area)
                largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                
                # Skip if face alignment failed
                if largest_face.embedding is None or largest_face.normed_embedding is None:
                    if not self.silent:
                        print(f"Warning: Face alignment failed for {filename}, skipping")
                    continue
                
                # Use normed_embedding instead of embedding to ensure alignment is used
                embedding = largest_face.normed_embedding
                self.reference_embeddings.append(embedding)
                
                # Save cropped reference face for verification
                face_crop = self._crop_face(img, largest_face.bbox)
                crop_filename = f"reference_{os.path.splitext(filename)[0]}.jpg"
                crop_path = os.path.join('crop', crop_filename)
                cv2.imwrite(crop_path, face_crop)
                
                processed_count += 1
                if not self.silent:
                    print(f"Processed reference face: {filename} -> {crop_filename}")
                
            except Exception as e:
                if not self.silent:
                    print(f"Error processing {filename}: {str(e)}")
                continue
        
        if self.reference_embeddings:
            self.reference_faces_processed = True
            
            # Save to cache
            self._save_to_cache()
            
            if not self.silent:
                print(f"Successfully processed {processed_count} reference faces")
                print(f"Stored {len(self.reference_embeddings)} reference embeddings")
        else:
            if not self.silent:
                print("Error: No reference faces could be processed")
    
    def _crop_face(self, img: np.ndarray, bbox: List[float], margin: float = 0.2) -> np.ndarray:
        """Crop face from image with margin"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Calculate margins
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Apply margins with bounds checking
        x1_crop = max(0, x1 - margin_x)
        y1_crop = max(0, y1 - margin_y)
        x2_crop = min(img.shape[1], x2 + margin_x)
        y2_crop = min(img.shape[0], y2 + margin_y)
        
        return img[y1_crop:y2_crop, x1_crop:x2_crop]
    
    def _encode_image_to_base64(self, img: np.ndarray) -> str:
        """Encode OpenCV image to base64 string"""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Encode to base64
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def analyze_face_in_image(self, img: np.ndarray, image_id: str) -> dict:
        """Analyze face in the given image and return results"""
        try:
            if not self.reference_faces_processed or not self.reference_embeddings:
                return {
                    "success": False,
                    "error": "Reference faces not properly loaded. Check reference_face/ directory."
                }
            
            # Detect faces in the image
            faces = self.app.get(img)
            
            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "No faces detected in the image",
                    "total_faces_detected": 0
                }
            
            # Get the largest face (most prominent)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Skip if face alignment failed
            if largest_face.embedding is None or largest_face.normed_embedding is None:
                return {
                    "success": False,
                    "error": "Face alignment failed for the detected face",
                    "total_faces_detected": len(faces)
                }
            
            # Crop the face
            face_crop = self._crop_face(img, largest_face.bbox)
            
            # Save cropped face to crop directory for verification
            crop_filename = f"generated_{image_id}.jpg"
            crop_path = os.path.join('crop', crop_filename)
            cv2.imwrite(crop_path, face_crop)
            
            # Encode face crop to base64
            face_crop_base64 = self._encode_image_to_base64(face_crop)
            
            # Use normed_embedding instead of embedding to ensure alignment is used
            face_embedding = largest_face.normed_embedding
            
            # Calculate similarity with all reference embeddings
            similarities = cosine_similarity(
                [face_embedding], 
                self.reference_embeddings
            )[0]
            
            # Get the highest similarity score
            max_similarity = float(np.max(similarities))
            max_similarity_index = int(np.argmax(similarities))
            
            score = np.mean(similarities)
            
            return {
                "success": True,
                "face_crop_base64": face_crop_base64,
                "similarity_score": score,#similarities[max_similarity_index],
                "crop_saved_path": crop_path,
                "face_bbox": largest_face.bbox.tolist(),
                "total_faces_detected": len(faces)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Face analysis failed: {str(e)}"
            }


def load_image_from_base64(base64_string: str) -> np.ndarray:
    """Load image from base64 string"""
    # Decode base64
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image from base64")
    
    return img


if __name__ == "__main__":
    # Test the service (verbose mode)
    print("Initializing Face Analysis Service...")
    service = FaceAnalysisService(silent=False)
    
    if service.reference_faces_processed:
        print("✅ Service initialized successfully")
        print(f"Number of reference embeddings: {len(service.reference_embeddings)}")
        print("Check 'crop/' directory for reference face crops")
    else:
        print("❌ Service initialization failed - no reference faces processed") 