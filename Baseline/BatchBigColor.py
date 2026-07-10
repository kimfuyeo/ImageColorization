import os
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path

# --- DIRECTORIES ---
input_dir = Path("Replace with your input directory")
final_output_dir = Path("Replace with your output directory")
comp_dir = Path("Replace with another directory for side by side comparison")

# The root of the BigColor repository
repo_root = Path("Replace with the original repo directory")

final_output_dir.mkdir(parents=True, exist_ok=True)
comp_dir.mkdir(parents=True, exist_ok=True)

# 1. THE INVISIBILITY CLOAK: Physically hide the GPU from PyTorch
my_env = os.environ.copy()
my_env["CUDA_VISIBLE_DEVICES"] = ""

# 2. HARDCODE the native CPU arguments
args = [
    "python", "-W", "ignore", "colorize_real.py",
    "--path_ckpt", "ckpts/bigcolor",
    "--epoch", "11",
    "--path_input", str(input_dir),
    "--device", "cpu"
]

print("Starting BigColor Batch Inference on Ryzen 9800X3D CPU...")
print(f"Executing: {' '.join(args)}\n")

# 3. Run the model with the hidden GPU environment
subprocess.run(args, cwd=repo_root, env=my_env)

# 4. Move generated images to your Shared Drive
out_dir = repo_root / "results_real"

if out_dir.exists() and any(out_dir.iterdir()):
    print(f"\n Moving generated images from {out_dir} to your Shared folder...")
    for img_file in out_dir.rglob("*"):
        if img_file.is_file() and img_file.suffix in ['.jpg', '.png', '.jpeg', '.JPG', '.JPEG']:
            shutil.copy2(str(img_file), str(final_output_dir / img_file.name))
            img_file.unlink() # Clean up original output
    print(f"All BigColor outputs saved to: {final_output_dir}")
else:
    print("\nInference finished, but couldn't locate outputs in results_real.")

# 5. Generate Side-by-Side Comparisons
print("\nStitching Side-by-Side Comparisons...")
output_files = list(final_output_dir.glob("*"))

if not output_files:
    print(f"No outputs found in {final_output_dir} to compare!")
else:
    for out_path in output_files:
        orig_found = False
        orig_path = None
        
        # Check against common image extensions
        for ext in ['.jpg', '.JPEG', '.png', '.JPG']:
            temp_path = input_dir / f"{out_path.stem}{ext}"
            if temp_path.exists():
                orig_path = temp_path
                orig_found = True
                break
                
        if not orig_found:
            print(f"Could not find original input for {out_path.name}")
            continue

        # Load both images
        img_orig = cv2.imread(str(orig_path))
        img_out = cv2.imread(str(out_path))

        if img_orig is None or img_out is None:
            continue

        # Force dimensions to match perfectly for stitching
        if img_orig.shape[:2] != img_out.shape[:2]:
            img_out = cv2.resize(img_out, (img_orig.shape[1], img_orig.shape[0]))

        # Concatenate horizontally (Original on Left, BigColor on Right)
        comparison = np.hstack((img_orig, img_out))
        
        save_path = comp_dir / f"{out_path.stem}_comparison.jpg"
        cv2.imwrite(str(save_path), comparison)
        
    print(f"All side-by-side comparisons saved to: {comp_dir}")
    print("Pipeline Complete!")
