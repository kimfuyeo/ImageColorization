import os
import subprocess
from pathlib import Path

# ==============================================================================
# 1. PIPELINE CONFIGURATION
# ==============================================================================
input_dir = Path("./input_data") 
ddcolor_out_dir = Path("./output/ddcolor_oracle")
unicolor_out_dir = Path("./output/unicolor_oracle")

python_cmd = "python3"
ddcolor_script = "./src/wrapper_ddcolor.py"
unicolor_script = "./src/wrapper_unicolor.py"

dd_task_file = Path("./temp/dd_tasks.txt")
uni_task_file = Path("./temp/uni_tasks.txt")

ddcolor_out_dir.mkdir(parents=True, exist_ok=True)
unicolor_out_dir.mkdir(parents=True, exist_ok=True)
dd_task_file.parent.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. BUILD THE ORACLE TASK LISTS
# ==============================================================================
valid_extensions = {".jpg", ".jpeg", ".png"}
input_images = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]

print(f"Building Oracle Tasks for {len(input_images)} images...")

dd_tasks = [] 
uni_tasks = [] 

for img_path in input_images:
    filename = img_path.name
    dd_output_path = ddcolor_out_dir / filename
    uni_output_path = unicolor_out_dir / filename
    
    # DDColor is blind: Input -> Output
    if not dd_output_path.exists():
        dd_tasks.append(f"{img_path}|{dd_output_path}")

    # UniColor Oracle: Input -> Ground Truth Exemplar -> Output
    if not uni_output_path.exists():
        uni_tasks.append(f"{img_path}|{img_path}|{uni_output_path}")

with open(dd_task_file, "w") as f: f.write("\n".join(dd_tasks))
with open(uni_task_file, "w") as f: f.write("\n".join(uni_tasks))

# ==============================================================================
# 3. EXECUTE THE BATCHES
# ==============================================================================
if dd_tasks:
    print(f"\nRunning DDColor Baseline on {len(dd_tasks)} images...")
    subprocess.run([python_cmd, ddcolor_script, "--task_file", str(dd_task_file)], check=True)
else:
    print("\nDDColor queue is empty. Outputs already exist.")

if uni_tasks:
    print(f"\nRunning UniColor ORACLE TEST on {len(uni_tasks)} images...")
    
    unicolor_project_dir = "./src/UniColor"
    sample_dir = f"{unicolor_project_dir}/sample"
    framework_dir = f"{unicolor_project_dir}/framework" 
    
    unicolor_env = os.environ.copy()
    unicolor_env["PYTHONPATH"] = f"{unicolor_project_dir}:{sample_dir}:{framework_dir}:" + unicolor_env.get("PYTHONPATH", "")
    
    subprocess.run(
        [python_cmd, unicolor_script, "--task_file", str(uni_task_file)], 
        cwd=unicolor_project_dir, 
        env=unicolor_env, 
        check=True
    )
else:
    print("\nUniColor queue is empty. Outputs already exist.")
