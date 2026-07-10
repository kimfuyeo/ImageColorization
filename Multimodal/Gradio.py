import gradio as gr
import os
import subprocess
from pathlib import Path
import re

# ==============================================================================
# 1. PIPELINE CONFIGURATION
# ==============================================================================
# Create a dedicated workspace for the UI to prevent clutter
WORKSPACE = Path("./temp/ui_workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

python_cmd = "python3"
ddcolor_script = "./src/wrapper_ddcolor.py"
unicolor_script = "./src/wrapper_unicolor.py"

ddcolor_dir = "./src/DDColor"
unicolor_dir = "./src/UniColor"

task_file = WORKSPACE / "current_ui_task.txt"

# ==============================================================================
# 2. THE PROCESSING ENGINE (WITH CRASH DETECTION)
# ==============================================================================
def process_image(target_img, exemplar_img, mode):
    if target_img is None:
        return None, "Please upload a Target Image."

    final_output = WORKSPACE / "final_output.png"
    dd_intermediate = WORKSPACE / "dd_intermediate.png"
    
    if final_output.exists(): final_output.unlink()
    if dd_intermediate.exists(): dd_intermediate.unlink()

    print(f"\nStarting UI Request: {mode}")

    def extract_time(output_string):
        match = re.search(r'INFERENCE_TIME:([0-9.]+)', output_string)
        if match:
            return float(match.group(1))
        return 0.0

    # ---------------------------------------------------------
    # MODE 1: Automated Baseline (DDColor)
    # ---------------------------------------------------------
    if mode == "Default Automated (DDColor)":
        with open(task_file, "w") as f:
            f.write(f"{target_img}|{final_output}")
            
        result = subprocess.run([python_cmd, ddcolor_script, "--task_file", str(task_file)], cwd=ddcolor_dir, capture_output=True, text=True)
        time_taken = extract_time(result.stdout)
        
        if time_taken == 0.0:
            return None, f"DDCOLOR CRASHED:\n{result.stderr}\n\n{result.stdout}"
            
        return str(final_output), f"DDColor complete. Inference Time: {time_taken:.4f} seconds."

    # ---------------------------------------------------------
    # MODE 2: Category Exemplar (UniColor)
    # ---------------------------------------------------------
    elif mode == "Exemplar Guided (UniColor)":
        if exemplar_img is None:
            return None, "UniColor requires an Exemplar Image!"
            
        with open(task_file, "w") as f:
            f.write(f"{target_img}|{exemplar_img}|{final_output}")
            
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{unicolor_dir}:{unicolor_dir}/sample:{unicolor_dir}/framework:" + env.get("PYTHONPATH", "")
        
        result = subprocess.run([python_cmd, unicolor_script, "--task_file", str(task_file)], cwd=unicolor_dir, env=env, capture_output=True, text=True)
        time_taken = extract_time(result.stdout)
        
        if time_taken == 0.0:
            return None, f"UNICOLOR CRASHED:\n{result.stderr}\n\n{result.stdout}"
            
        return str(final_output), f"UniColor complete. Inference Time: {time_taken:.4f} seconds."

    # ---------------------------------------------------------
    # MODE 3: Cascaded (DDColor Base -> UniColor)
    # ---------------------------------------------------------
    elif mode == "Cascaded Pipeline (DDColor Base -> UniColor)":
        with open(task_file, "w") as f:
            f.write(f"{target_img}|{dd_intermediate}")
        
        dd_result = subprocess.run([python_cmd, ddcolor_script, "--task_file", str(task_file)], cwd=ddcolor_dir, capture_output=True, text=True)
        dd_time = extract_time(dd_result.stdout)
        
        if dd_time == 0.0:
            return None, f"DDCOLOR CRASHED IN STAGE 1:\n{dd_result.stderr}\n\n{dd_result.stdout}"
        
        with open(task_file, "w") as f:
            f.write(f"{target_img}|{dd_intermediate}|{final_output}")
            
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{unicolor_dir}:{unicolor_dir}/sample:{unicolor_dir}/framework:" + env.get("PYTHONPATH", "")
        
        uc_result = subprocess.run([python_cmd, unicolor_script, "--task_file", str(task_file)], cwd=unicolor_dir, env=env, capture_output=True, text=True)
        uc_time = extract_time(uc_result.stdout)
        
        if uc_time == 0.0:
            return None, f"UNICOLOR CRASHED IN STAGE 2:\n{uc_result.stderr}\n\n{uc_result.stdout}"
            
        total_time = dd_time + uc_time
        return str(final_output), f"Cascaded Pipeline complete. Total Time: {total_time:.4f}s (DD: {dd_time:.4f}s + UC: {uc_time:.4f}s)."
    
# ==============================================================================
# 3. GRADIO USER INTERFACE LAYOUT
# ==============================================================================
theme = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(theme=theme, title="Multimodal Colorization Pipeline") as demo:
    gr.Markdown("# Multimodal Image Colorization System")
    gr.Markdown("Upload a grayscale target, select your pipeline strategy, and generate a colorized output.")
    
    with gr.Row():
        # Input Column
        with gr.Column():
            target_input = gr.Image(type="filepath", label="Target Image (Grayscale Input)")
            exemplar_input = gr.Image(type="filepath", label="Exemplar Image (Optional Reference)")
            
            mode_selector = gr.Radio(
                choices=[
                    "Default Automated (DDColor)", 
                    "Exemplar Guided (UniColor)", 
                    "Cascaded Pipeline (DDColor Base -> UniColor)"
                ],
                value="Default Automated (DDColor)",
                label="Colorization Mode",
                info="Select the neural network architecture to process this image."
            )
            
            submit_btn = gr.Button("Run Colorization", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        # Output Column
        with gr.Column():
            result_output = gr.Image(type="filepath", label="Final Colorized Output", interactive=False)

    # Wire the button to the python function
    submit_btn.click(
        fn=process_image,
        inputs=[target_input, exemplar_input, mode_selector],
        outputs=[result_output, status_text]
    )

if __name__ == "__main__":
    print("Launching UI... Check your terminal for the local web address!")
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[str(WORKSPACE)])
