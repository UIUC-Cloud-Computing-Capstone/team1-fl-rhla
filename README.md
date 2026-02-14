# 👋 Federated Learning Resource-Aware Heterogeneous LoRA Allocation

## 🧪 Getting Started

### ⚙️ Environment Setup

Ensure you have [Conda](https://docs.conda.io/) installed. 

Then activate the environment using the provided file:

Default:

```bash
conda env create --name env.fl --file=setup/environment-final.yml
conda activate env.fl
```

### 🚀 Running the Code

First, set up the HuggingFace Accelerate configuration:

(Optional) If your system uses multiple GPUs, uncomment the "Multiple GPUs" config options in `setup/accelerate_default_config.yaml`

Copy the accelerate config to the huggingface directory: 

```bash
mkdir -p ~/.cache/huggingface/accelerate/
cp setup/accelerate_default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

Next, launch the training script for the CIFAR-100 dataset:

To experiment the Rank Estimator, run the command:
```bash
# Default config: config/rank_estimator_ours.yaml
python scripts/run_rank_estimation.py

# Or with an explicit config path
python scripts/run_rank_estimation.py config/rank_estimator_ours.yaml
```
Using an NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0), the Rank Estimator takes approximately 50 seconds to complete. The results are logged in the terminal upon completion:
```
...
Per client (total rank budget): [288, 432, 576]
Total time to finish the rank estimation task: 50.61s
``` 

To run memory breakdown comparison (estimated vs profiled) and generate LaTeX tables:

```bash
python scripts/run_memory_breakdown_comparison.py qv
# Or
python scripts/run_memory_breakdown_comparison.py qv --config config/memory_breakdown_comparison.yaml
```

To utilize the rank budget to fine-tune the model, run one of the scripts in `scripts/`. For example, our method on CIFAR-100:

```bash
bash scripts/run-cifar100-Ours.sh
```

Other experiment scripts (CIFAR-100, LEDGAR, IID/non-IID, ablations, baselines) are in `scripts/`; run any `run-*.sh` from the project root.

---

## 📁 Project Structure

```
.
├── algorithms/
│   ├── engine/   # Federated learning coordination logic
│   └── solver/   # Local training procedures
├── config/         # YAML configuration files
├── data/           # Dataset cache directory
├── figures/        # Generated PDF figures (from scripts/figures/*.py)
├── log/            # Training and experiment logs (created at runtime)
├── models/         # Model layer reference files (BERTlayerName, ViTLayerName)
├── setup/         # Environment and tool config (environment-final.yml, accelerate_default_config.yaml)
├── scripts/        # Run and figure scripts
│   ├── figures/    # Scripts to generate figures
│   ├── run_rank_estimation.py
│   ├── run_memory_breakdown_comparison.py
│   └── run-*.sh    # Experiment scripts (CIFAR-100, LEDGAR, baselines, ablations)
├── utils/          # Utility functions (including estimator.py for rank estimation)
├── main.py         # Entry point for training
└── test.py         # Evaluation and testing routines
```

---

## 📄 Citation

If you find this work useful for your research, please cite papers
```
@article{xie2026hralora,
  title={Heterogeneous-Resource-Aware Federated Learning with Intelligent LoRA Allocation and Aggregation},
  author={Xie, Youye and Lian, Yao and Chen, Kevin and Latif, Abdul and Zhao, Lingzhi and Farivar, Reza},
  journal={},
  year={2026}
}
@article{zhang2025fed,
  title={Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation},
  author={Zhang, Zikai and Liu, Ping and Xu, Jiahao and Hu, Rui},
  journal={arXiv preprint arXiv:2506.12213},
  year={2025}
}
```

---

## References

This is a fork from the official code for the paper accepted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

```bibtext
@article{zhang2025fed,
  title={Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation},
  author={Zhang, Zikai and Liu, Ping and Xu, Jiahao and Hu, Rui},
  journal={arXiv preprint arXiv:2506.12213},
  year={2025}
}
```

---

## 📬 Contact

For any questions or suggestions, please feel free to open an issue on this repository or contact the authors directly.
