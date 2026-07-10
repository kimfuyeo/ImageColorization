import time
import torch
import cv2
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm
from modelscope.outputs import OutputKeys

parser = argparse.ArgumentParser()
parser.add_argument("--task_file", required=True)
args = parser.parse_args()

device_name = 'cuda'
print("\n[DDColor] CUDA Active. Firing up the GPU...")

local_model_path = "./weights/cv_ddcolor_image-colorization"

print(f"[DDColor] Loading offline weights from: {local_model_path}")
try:
    img_colorization = pipeline(Tasks.image_colorization, model=local_model_path, device=device_name)
except Exception as e:
    print(f"\n[DDColor] Failed to load offline model: {e}")
    exit(1)

with open(args.task_file, "r") as f:
    tasks = [line.strip() for line in f.readlines() if line.strip()]

for task in tqdm(tasks, desc="DDColor Batching"):
    in_path, out_path = task.split("|")
    try:
        # === TIMING START ===
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = img_colorization(in_path)
        
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time
        
        # --- THE LOGGER ---
        with open('./logs/inference_metrics.csv', 'a') as log_f:
            log_f.write(f"DDColor,{in_path},{inference_time:.4f}\n")
        # === TIMING END ===

        cv2.imwrite(out_path, result[OutputKeys.OUTPUT_IMG])
        
        # Broadcast the exact time for the Gradio UI to catch
        print(f"\nINFERENCE_TIME:{inference_time:.4f}")
        
    except Exception as e:
        print(f"\n[DDColor] Crash on {in_path}: {e}")

print("\n[DDColor] Batch complete!")
