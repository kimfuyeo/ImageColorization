import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- CONFIG ---
base_dir = Path("./temporal_workspace")
ebsynth_bin = Path("./ebsynth/bin/ebsynth")
video_frames = base_dir / "video_frames"
keys_gray = base_dir / "keyframes_gray"
keys_color = base_dir / "keyframes_color"
output_dir = base_dir / "Final_Video_Frames"
output_dir.mkdir(exist_ok=True)

MAX_WORKERS = 12

def process_frame(current_key, target_frame):
    """The actual heavy lifting command for EBSynth"""
    cmd = [
        str(ebsynth_bin),
        "-style", str(keys_color / current_key),
        "-guide", str(keys_gray / current_key), str(video_frames / target_frame),
        "-output", str(output_dir / target_frame)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_frame_wrapper(params):
    """The bridge for the ProcessPoolExecutor"""
    return process_frame(*params)

if __name__ == "__main__":
    all_frames = sorted([f.name for f in video_frames.glob("*.png")])
    color_keys = sorted([f.name for f in keys_color.glob("*.png")])
    
    if not color_keys:
        print("Error: No colorized keyframes found! Run Script 2 first.")
        exit()

    tasks = []
    for i in range(len(color_keys)):
        current_key = color_keys[i]
        start_idx = int(current_key.split('.')[0])
        
        # Determine where this scene ends (either the next keyframe, or end of video)
        end_idx = int(color_keys[i+1].split('.')[0]) if i + 1 < len(color_keys) else len(all_frames)
        
        for frame_idx in range(start_idx, end_idx):
            tasks.append((current_key, f"{frame_idx:05d}.png"))

    print(f"Starting Parallel Synthesis...")
    print(f"Tasks: {len(tasks)} frames | Threads: {MAX_WORKERS}")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_frame_wrapper, tasks), total=len(tasks)))

    print(f"\nSTAGE 3 COMPLETE! Final sequence ready in: {output_dir}")
    print("\nTo stitch frames into final video, run:")
    print("ffmpeg -framerate 30 -i %05d.png -c:v h264_nvenc -preset p6 -b:v 10M output_video.mp4")
