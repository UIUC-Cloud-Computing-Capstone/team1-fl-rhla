import json
import yaml
import os
import shutil

def apply_json_to_yaml(json_path, yaml_path):
    # Load JSON list
    with open(json_path, 'r') as jf:
        json_list = json.load(jf)
        if not isinstance(json_list, list):
            raise ValueError("JSON file must contain a list of objects")

    # Load base YAML
    with open(yaml_path, 'r') as yf:
        base_yaml = yaml.safe_load(yf)

    # Ensure temp directory exists
    temp_dir = "./config/experiments/cifar100_vit_lora/temp"
    os.makedirs(temp_dir, exist_ok=True)

    # For each object in the JSON list, create a modified YAML copy
    for i, json_obj in enumerate(json_list):
        if not isinstance(json_obj, dict):
            raise ValueError(f"JSON entry at index {i-1} is not an object")

        # Make a copy of the YAML data
        new_yaml = base_yaml.copy()

        # Replace matching fields
        for key, value in json_obj.items():
            if key in new_yaml:
                new_yaml[key] = value

        # Define output path
        filename = f"config_{i}.yaml"
        output_path = os.path.join(temp_dir, filename)

        # Write the new YAML file
        with open(output_path, 'w') as out:
            yaml.dump(new_yaml, out, sort_keys=False)

        print(f"Created {output_path}")

if __name__ == "__main__":
    # Example usage
    apply_json_to_yaml("./multirun/experiments.json", "./config/experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50.yaml")
