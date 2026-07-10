import os
import sys
import argparse
import torch
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# --- FIX PYTHON PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(current_dir, 'sample'))

import sample.ImageMatch.utils.util as broken_util
broken_util.np = np
# ====================================================================

from sample.colorizer import Colorizer

# Allowlist lightning checkpoint for PyTorch 2.6 security
torch.serialization.add_safe_globals([ModelCheckpoint])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt_file = './weights/unicolor_imagenet/mscoco_step259999.ckpt'
    if not os.path.exists(ckpt_file):
        ckpt_file = ckpt_file.replace('.ckpt', '')

    print("\n[UniColor] Loading MSCOCO weights into VRAM...")
    colorizer = Colorizer(ckpt_file, device, [256, 256], load_clip=True, load_warper=True)

    with open(args.task_file, "r") as f:
        tasks = [line.strip() for line in f.readlines() if line.strip()]

    for task in tqdm(tasks, desc="UniColor Batching"):
        parts = task.split("|")
        
        if len(parts) == 3:
            in_path, ex_path, out_path = parts
        else:
            in_path, out_path = parts
            ex_path = in_path
            
        try:
            I_gray = Image.open(in_path).convert('RGB')
            I_exp = Image.open(ex_path).convert('RGB')
            
            # === TIMING START ===
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            points, warped = colorizer.get_strokes_from_exemplar(I_gray, I_exp)
            out_img = colorizer.sample(I_gray, points, topk=100)
            
            # === TIMING END ===
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time
            
            # --- THE LOGGER ---
            with open('./logs/inference_metrics.csv', 'a') as log_f:
                log_f.write(f"UniColor,{in_path},{inference_time:.4f}\n")
            print(f"INFERENCE_TIME:{inference_time:.4f}")
            
            if isinstance(out_img, np.ndarray):
                out_img = Image.fromarray(out_img)
            out_img.save(out_path)
            
        except Exception as e:
            print(f"\n[UniColor] Crash on {in_path}: {e}")

if __name__ == "__main__":
    main()
