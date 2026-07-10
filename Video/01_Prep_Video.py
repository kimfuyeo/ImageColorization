import os
import cv2
from pathlib import Path
from tqdm import tqdm
from scenedetect import detect, ContentDetector

# --- CONFIG ---
video_path = "./input_video.mp4"
base_dir = Path("./temporal_workspace")

video_frames_dir = base_dir / "video_frames"
keyframes_gray_dir = base_dir / "keyframes_gray"

video_frames_dir.mkdir(parents=True, exist_ok=True)
keyframes_gray_dir.mkdir(parents=True, exist_ok=True)

FORCED_ANCHOR_INTERVAL = 60 

# --- 1. EXTRACT ALL FRAMES (AS GRAYSCALE) ---
print(f"Opening Video: {video_path}")
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_idx in tqdm(range(total_frames), desc="Extracting Full Video"):
    ret, frame = cap.read()
    if not ret: break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filename = f"{frame_idx:05d}.png"
    cv2.imwrite(str(video_frames_dir / filename), gray_frame)

cap.release()

# --- 2. DETECT SCENES & INJECT TEMPORAL ANCHORS ---
print(f"\nScanning for camera cuts and injecting temporal anchors...")
scene_list = detect(video_path, ContentDetector(threshold=15, luma_only=True))

target_keyframes = set()

# A. Add the hard scene cuts
for scene in scene_list:
    target_keyframes.add(scene[0].get_frames())

# B. Add the forced interval anchors to prevent long-pan bleeding
for frame_idx in range(0, total_frames, FORCED_ANCHOR_INTERVAL):
    target_keyframes.add(frame_idx)

# Sort them so they process in chronological order
sorted_keyframes = sorted(list(target_keyframes))
print(f"Generated {len(sorted_keyframes)} total tracking keyframes.")

for frame_idx in sorted_keyframes:
    filename = f"{frame_idx:05d}.png"
    source_path = video_frames_dir / filename
    target_path = keyframes_gray_dir / filename
    
    if source_path.exists():
        img = cv2.imread(str(source_path))
        cv2.imwrite(str(target_path), img)

print("\nSTAGE 1 COMPLETE! Gray frames and tracking anchors are ready.")
