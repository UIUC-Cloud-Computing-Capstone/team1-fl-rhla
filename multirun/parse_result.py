import os
import re

def getAccuracyList(file_path):
    """Extract all test_acc values from a log file as floats."""
    with open(file_path, "r") as f:
        text = f.read()
    test_acc_values = re.findall(r'test_acc\s*=\s*([\d.]+)', text)
    return [float(x) for x in test_acc_values]


def main():
    base_dir = "./log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg"
    output_dir = "./multirun/results"
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        if "exp_log.txt" not in files:
            continue

        exp_log_path = os.path.join(root, "exp_log.txt")

        with open(exp_log_path, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            print(f"Skipping {exp_log_path} (missing required lines)")
            continue

        # Extract timestamp from first line
        line1 = lines[0].strip()
        match_timestamp = re.search(r'(_\d{4}-\d{2}-\d{2}_[\d-]+)', line1)
        if not match_timestamp:
            print(f"⚠️ Could not extract timestamp in {exp_log_path}")
            continue
        timestamp = match_timestamp.group(1)

        # Extract config name from second line
        line2 = lines[1].strip()
        match_config = re.search(r'config[^/]*?\.yaml', line2)
        if not match_config:
            print(f"⚠️ Could not extract config name in {exp_log_path}")
            continue

        config_name_yaml = match_config.group(0)  # e.g. config_1.yaml
        config_name = os.path.splitext(config_name_yaml)[0]  # remove .yaml

        # Get test accuracy list
        test_acc_values = getAccuracyList(exp_log_path)

        # Write output to ./multirun/results/{timestamp}_{config_name}.txt
        output_filename = f"{timestamp}_{config_name}.txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w") as out:
            if test_acc_values:
                out.write("\n".join(map(str, test_acc_values)))

        print(f"✅ Saved {len(test_acc_values)} accuracy values → {output_path}")


if __name__ == "__main__":
    main()
