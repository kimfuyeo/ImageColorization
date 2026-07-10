import os
import torch
import functools
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

# --- 1. PYTORCH 2.6 SECURITY BYPASS ---
torch.serialization.add_safe_globals([functools.partial])
import fastai.basic_train
original_load = torch.load

def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = safe_load

# --- 2. INITIALIZE GPU & MODEL ---
from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)

from deoldify.visualize import get_image_colorizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading DeOldify Model into GPU...")
colorizer = get_image_colorizer(artistic=False)
print("Model Loaded Successfully!")

# --- 3. BATCH PROCESSING SETUP ---
input_dir = Path("./input_data")
output_dir = Path("./output/deoldify_stable_raw")
comparison_dir = Path("./output/comparisons_deoldify") 

output_dir.mkdir(parents=True, exist_ok=True)
comparison_dir.mkdir(parents=True, exist_ok=True)

# (10-21 = Vibrant, 35-45 = Safe/Stable)
render_factor = 35  

image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
print(f"Found {len(image_files)} images. Starting processing at Render Factor {render_factor}...")

# --- 4. THE INFERENCE LOOP ---
for img_path in tqdm(image_files, desc=f"RF {render_factor} Colorizing"):
    save_path = output_dir / f"{img_path.stem}_DeOldify_RF{render_factor}.png"
    comp_path = comparison_dir / f"{img_path.stem}_Comparison_RF{render_factor}.png"
    
    if save_path.exists() and comp_path.exists():
        continue # Skip if already processed
        
    try:
        # Run Inference
        result_img = colorizer.get_transformed_image(path=str(img_path), render_factor=render_factor)
        
        if result_img is not None:
            # Save Pure Output
            result_img.save(str(save_path))
            
            # Create and Save Comparison
            orig_img = Image.open(img_path).convert("RGB")
            orig_resized = orig_img.resize(result_img.size)
            combo = Image.new('RGB', (result_img.width * 2, result_img.height))
            combo.paste(orig_resized, (0, 0))
            combo.paste(result_img, (result_img.width, 0))
            combo.save(str(comp_path))
            
    except Exception as e:
        print(f"Failed to process {img_path.name}: {e}")

print(f"\nAll RF {render_factor} generations complete!")
