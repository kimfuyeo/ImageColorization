import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- PYTORCH LIGHTNING SECURITY BYPASS ---
import pytorch_lightning
torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])

# --- PATH INJECTION FOR UNICOLOR ---
unicolor_project_dir = "./src/UniColor"
if unicolor_project_dir not in sys.path:
    sys.path.insert(0, unicolor_project_dir)
    sys.path.insert(0, os.path.join(unicolor_project_dir, 'sample'))

try:
    from sample.colorizer import Colorizer
except ImportError as e:
    print(f"Failed to import UniColor. Check your paths: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
base_dir = Path("./temporal_workspace")
input_dir = base_dir / "keyframes_gray"
output_dir = base_dir / "keyframes_color"
reference_path = base_dir / "reference_exemplar.jpg"  # Target chromatic guide
output_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. LOAD THE REFERENCE IMAGE ---
if not reference_path.exists():
    print(f"Error: Could not find reference image at {reference_path}")
    sys.exit(1)

print(f"Loading Exemplar Reference: {reference_path.name}")
exemplar_img = Image.open(str(reference_path)).convert('RGB')

# --- 2. INITIALIZE UNICOLOR ---
print("Loading UniColor into VRAM...")
unicolor_ckpt = './weights/unicolor_imagenet/mscoco_step259999.ckpt'
colorizer = Colorizer(unicolor_ckpt, device, [256, 256], load_clip=True, load_warper=True)

# --- 3. BATCH PROCESSING LOOP ---
frames = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == '.png'])
print(f"Injecting Exemplar Palette into {len(frames)} keyframes...")

with torch.inference_mode(): # Block memory leaks
    for frame_path in tqdm(frames, desc="Colorizing"):
        out_path = output_dir / frame_path.name
        
        if out_path.exists():
            continue
            
        try:
            # Open the grayscale keyframe structure
            target_img = Image.open(str(frame_path)).convert('RGB')
            
            # Run UniColor using the static photo as the chromatic guide
            final_array = colorizer.sample(
                image=target_img,
                strokes=[],        
                topk=100,          
                prior_image=exemplar_img 
            )
            
            # Save the final keyframe
            if isinstance(final_array, np.ndarray):
                final_image = Image.fromarray(final_array)
            else:
                final_image = final_array
                
            final_image.save(str(out_path))
            
            # Clean up frame-specific memory
            del target_img
            del final_image
                
        except Exception as e:
            print(f"\nPipeline crashed on {frame_path.name}: {e}")

print(f"\nSTAGE 2 COMPLETE! Exemplar-guided keyframes saved to: {output_dir}")
