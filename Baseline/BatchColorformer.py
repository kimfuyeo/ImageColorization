import os
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path

# --- DIRECTORIES ---
input_dir = Path("./input")
temp_output_dir = Path("./temp_output") 
final_output_dir = Path("./output")
comp_dir = Path("./comparisons")

final_output_dir.mkdir(parents=True, exist_ok=True)
comp_dir.mkdir(parents=True, exist_ok=True)

# 1. Safety Check for Weights
weight_path = Path("./pretrain/net_g_200000.pth")
if not weight_path.exists():
    print("Error: Model weights not found at 'pretrain/net_g_200000.pth'!")
    print("Please check the ColorFormer GitHub README, download the file, and place it in the 'pretrain' folder.")
    exit(1)

# 2. Execution Arguments
args = [
    "python3", "inference/inference_colorformer.py",
    "--input", str(input_dir),
    "--output", str(temp_output_dir),
    "--model_path", str(weight_path)
]

print("Starting Colorformer Inference")
print(f"Executing: {' '.join(args)}\n")

# Run the Transformer
subprocess.run(args)

# 3. Move and Stitch Images
print("\nMoving and Stitching Comparisons...")
output_files = list(temp_output_dir.glob("*"))

if not output_files:
    print(f"No outputs found in {temp_output_dir}!")
else:
    success_count = 0
    for out_path in output_files:
        if not out_path.is_file(): 
            continue
        
        # Move safely to final output directory
        final_dest = final_output_dir / out_path.name
        shutil.copy2(str(out_path), str(final_dest))
        
        # Stitching Logic
        orig_found = False
        orig_path = None
        
        for ext in ['.jpg', '.JPEG', '.png', '.JPG']:
            temp_path = input_dir / f"{out_path.stem}{ext}"
            if temp_path.exists():
                orig_path = temp_path
                orig_found = True
                break
                
        if not orig_found:
            print(f"Could not find original input for {out_path.name}")
            continue

        img_orig = cv2.imread(str(orig_path))
        img_out = cv2.imread(str(final_dest))

        if img_orig is None or img_out is None:
            continue

        # Force dimensions to match
        if img_orig.shape[:2] != img_out.shape[:2]:
            img_out = cv2.resize(img_out, (img_orig.shape[1], img_orig.shape[0]))

        # Stitch Original | Colorformer side-by-side
        comparison = np.hstack((img_orig, img_out))
        save_path = comp_dir / f"{out_path.stem}_comparison.jpg"
        cv2.imwrite(str(save_path), comparison)
        success_count += 1
        
    # Clean up the local temp directory
    shutil.rmtree(temp_output_dir)
    print(f"Successfully processed and stitched {success_count} images!")
    print(f"Comparisons saved to: {comp_dir}")
