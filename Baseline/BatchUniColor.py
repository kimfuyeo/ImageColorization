import os
import cv2
import torch
from tqdm import tqdm

# ==============================================================================
# 1. SETUP YOUR DIRECTORIES
# ==============================================================================
input_gt_dir = "./input_data"               # Folder with your 1000 original color images
output_dir = "./output/unicolor_oracle_raw" # For the raw AI outputs (used for PSNR math)
grid_dir = "./output/comparisons_oracle"    # For the side-by-side visual comparisons

os.makedirs(output_dir, exist_ok=True)
os.makedirs(grid_dir, exist_ok=True)        

# ==============================================================================
# 2. INITIALIZE UNICOLOR 
# ==============================================================================
print("Loading UniColor Model into VRAM...")
# NOTE: Replace this section with the exact import/loading code from your UniColor repo.
# Example: 
# from models.unicolor_pipeline import UniColorModel
# model = UniColorModel(device='cuda')
# model.load_weights('checkpoints/unicolor_fp16.pth')

valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(input_gt_dir) if f.lower().endswith(valid_extensions)]

print(f"Found {len(image_files)} images for Oracle Testing. Starting batch...\n")

# ==============================================================================
# 3. THE INFERENCE & STITCHING LOOP
# ==============================================================================
for filename in tqdm(image_files, desc="Processing Oracle Batch"):
    try:
        gt_path = os.path.join(input_gt_dir, filename)
        out_path = os.path.join(output_dir, filename)
        grid_path = os.path.join(grid_dir, f"grid_{filename}")
        
        # Skip if the grid already exists (useful if the script crashes halfway)
        if os.path.exists(grid_path):
            continue

        # Load the Ground Truth (GT) image
        gt_bgr = cv2.imread(gt_path)
        if gt_bgr is None:
            print(f"\nSkipping {filename}: Unreadable file.")
            continue
            
        # Create the grayscale input dynamically
        gray_1ch = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray_1ch, cv2.COLOR_GRAY2BGR)

        # ----------------------------------------------------------------------
        # RUN UNICOLOR INFERENCE HERE
        # ----------------------------------------------------------------------
        # Pass the grayscale as the target, and the original color as the exemplar.
        # Example pseudo-code:
        #
        # with torch.no_grad():
        #     output_tensor = model.colorize_with_exemplar(target_image=gray_3ch, exemplar_image=gt_bgr)
        # result_bgr = tensor_to_cv2(output_tensor)
        
        # NOTE: Delete this placeholder line once your model inference is slotted in
        result_bgr = gt_bgr 
        
        # Save the raw output for your quantitative math scripts
        cv2.imwrite(out_path, result_bgr)

        # ----------------------------------------------------------------------
        # THE SIDE-BY-SIDE STITCHING LOGIC
        # ----------------------------------------------------------------------
        # 1. Get exact dimensions of the Ground Truth
        target_height, target_width = gt_bgr.shape[:2]
        
        # 2. Resize the AI result to perfectly match the GT height (prevents OpenCV crash)
        result_resized = cv2.resize(result_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 3. Stitch them horizontally: [Original Ground Truth] | [UniColor Output]
        grid_image = cv2.hconcat([gt_bgr, result_resized])
        
        # 4. Save the side-by-side comparison
        cv2.imwrite(grid_path, grid_image)

    except Exception as e:
        print(f"\nError processing {filename}: {str(e)}")
        torch.cuda.empty_cache()

print("\nOracle Test Batch & Grid Generation Complete!")
print(f"Raw outputs saved to: {output_dir}")
print(f"Visual grids saved to: {grid_dir}")
