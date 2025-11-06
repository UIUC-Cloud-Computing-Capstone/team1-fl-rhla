import subprocess
import os
import glob

def main():
    # Step 1: Run config_merge.py to generate YAMLs
    print("Running config_merge.py...")
    subprocess.run(["python", "./multirun/config_merge.py"], check=True)
    
    # Step 2: Find all YAML files in ./temp/
    temp_dir = "./config/experiments/cifar100_vit_lora/temp"
    yaml_files = sorted(glob.glob(os.path.join(temp_dir, "*.yaml")))

    if not yaml_files:
        print("No YAML files found in ./temp/.")
        return

    # Step 3: For each YAML file, run the training command
    for yaml_path in yaml_files:
        corrected_path = f"./{yaml_path[9:]}" # Root in /config
        print(f"\nRunning accelerate launch with config: {corrected_path}")
        command = [
            "bash", "-c",
            f"NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG "
            f"accelerate launch --main_process_port 29505 main.py --config_name '{corrected_path}'"
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Command failed for {corrected_path}: {e}")

if __name__ == "__main__":
    main()
