import os
import cv2
import numpy as np
from pathlib import Path
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

# --- DIRECTORY SETUP ---
input_dir = Path("./input_data")
output_dir = Path("./output/ddcolor_raw")
comparison_dir = Path("./output/comparisons") # Folder for stitched images

# Create folders if they don't exist
output_dir.mkdir(parents=True, exist_ok=True)
comparison_dir.mkdir(parents=True, exist_ok=True)

print("Loading DDColor Model inside Docker Container...")
# Automatically fetches the DDColor weights from ModelScope

print("DDColor Pipeline Ready!\n")
# Ensure the model path points to the generic local directory for GitHub users
colorizer = pipeline('image-colorization', model='./offline_model', device='cpu')

# Grab all common image formats
image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPEG")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))

if not image_files:
    print(f"No images found in {input_dir}!")
else:
    print(f"Found {len(image_files)} images to colorize. Starting batch process...")

# --- BATCH PROCESSING LOOP ---
for i, img_path in enumerate(image_files, 1):
    print(f"Processing {i}/{len(image_files)}: {img_path.name}")
    
    try:
        # 1. Run DDColor Inference
        result = colorizer(str(img_path))
        colorized_img = result[OutputKeys.OUTPUT_IMG]
        
        # 2. Save the pure standalone output (Needed for quantitative analysis!)
        save_path = output_dir / f"{img_path.stem}_DDColor.png"
        cv2.imwrite(str(save_path), colorized_img)
        
        # 3. Create the Side-by-Side Comparison
        # Load the original image via OpenCV
        orig_img = cv2.imread(str(img_path))
        
        # Resize original to perfectly match the colorized output dimensions 
        h, w = colorized_img.shape[:2]
        orig_resized = cv2.resize(orig_img, (w, h))
        
        # Stitch them horizontally [Original Left | Colorized Right]
        stitched_img = cv2.hconcat([orig_resized, colorized_img])
        
        # Save the stitched comparison
        comp_path = comparison_dir / f"{img_path.stem}_comparison.png"
        cv2.imwrite(str(comp_path), stitched_img)
        
    except Exception as e:
        print(f"Failed to process {img_path.name}: {e}")

print("\nAll DDColor generations complete!")
print(f"Pure Outputs saved to: {output_dir}")
print(f"Comparisons saved to: {comparison_dir}")
