#!/usr/bin/env python
# coding=utf-8
# Script to create a DPO dataset for FLUX training

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import datasets
from datasets import Dataset, Features, Value, Image as ImageFeature
import numpy as np
from tqdm.auto import tqdm
import io

def parse_args():
    parser = argparse.ArgumentParser(description="Create a DPO dataset for FLUX training")
    parser.add_argument(
        "--preferred_images_dir",
        type=str,
        required=True,
        help="Directory containing preferred (better quality) images",
    )
    parser.add_argument(
        "--less_preferred_images_dir",
        type=str,
        required=True,
        help="Directory containing less preferred (lower quality) images",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        required=True,
        help="JSON file containing image to caption mappings",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="flux_dpo_dataset",
        help="Path to save the dataset",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Train/validation split ratio",
    )
    return parser.parse_args()

def load_image_bytes(image_path):
    """Load image and convert to bytes for dataset storage"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_dpo_dataset(args):
    # Load captions
    with open(args.captions_file, 'r') as f:
        captions = json.load(f)
    
    # Get image files
    preferred_images = sorted([f for f in os.listdir(args.preferred_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    less_preferred_images = sorted([f for f in os.listdir(args.less_preferred_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(preferred_images) != len(less_preferred_images):
        raise ValueError(f"Number of preferred ({len(preferred_images)}) and less preferred ({len(less_preferred_images)}) images must match")
    
    # Prepare dataset entries
    dataset_entries = []
    
    print("Creating dataset entries...")
    for pref_img, less_img in tqdm(zip(preferred_images, less_preferred_images), total=len(preferred_images)):
        # Get base filename without extension for caption lookup
        pref_base = os.path.splitext(pref_img)[0]
        
        if pref_base not in captions:
            print(f"Warning: No caption found for {pref_base}, skipping...")
            continue
        
        caption = captions[pref_base]
        
        # Load images as bytes
        pref_img_bytes = load_image_bytes(os.path.join(args.preferred_images_dir, pref_img))
        less_img_bytes = load_image_bytes(os.path.join(args.less_preferred_images_dir, less_img))
        
        if pref_img_bytes is None or less_img_bytes is None:
            print(f"Warning: Failed to load images for pair {pref_img}, {less_img}, skipping...")
            continue
        
        # Create entry with 50% chance of swapping order
        if np.random.random() < 0.5:
            entry = {
                'jpg_0': pref_img_bytes,
                'jpg_1': less_img_bytes,
                'label_0': 0,  # 0 means jpg_0 is preferred
                'caption': caption
            }
        else:
            entry = {
                'jpg_0': less_img_bytes,
                'jpg_1': pref_img_bytes,
                'label_0': 1,  # 1 means jpg_1 is preferred
                'caption': caption
            }
        
        dataset_entries.append(entry)
    
    if not dataset_entries:
        raise ValueError("No valid dataset entries were created. Please check your input data.")
    
    # Create dataset with explicit features
    features = Features({
        'jpg_0': Value('binary'),
        'jpg_1': Value('binary'),
        'label_0': Value('int64'),
        'caption': Value('string')
    })
    
    # Create dataset
    dataset = Dataset.from_list(dataset_entries, features=features)
    
    # Split dataset
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(train_size=args.split_ratio)
    
    # Save dataset
    os.makedirs(args.output_path, exist_ok=True)
    split_dataset.save_to_disk(args.output_path)
    
    print(f"\nDataset saved to {args.output_path}")
    print(f"Train set size: {len(split_dataset['train'])}")
    print(f"Test set size: {len(split_dataset['test'])}")
    print("\nDataset features:")
    print(split_dataset['train'].features)

def main():
    args = parse_args()
    create_dpo_dataset(args)

if __name__ == "__main__":
    main() 