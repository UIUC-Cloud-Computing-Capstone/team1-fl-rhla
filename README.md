# 👋 Heterogeneous-Resource-Aware Federated Learning with Intelligent LoRA Allocation and Aggregation (HRALoRA)

Federated learning (FL) enables privacy-preserving model training across distributed devices, while Low-Rank Adaptation (LoRA) reduces compute and communication. Heterogeneous client memory and network constraints, however, make it hard to choose LoRA ranks and aggregate updates from resource-diverse clients. **HRALoRA** tackles this end-to-end and matches or outperforms prior LoRA-based FL with about half the trainable parameters.

## Core Functionality

HRALoRA has two core components:

- **Rank Estimator:** Estimates each client’s LoRA rank budget from its GPU memory and network limits so every client can participate within its constraints.
- **Rank Utilizer:** Uses the rank budget to train and aggregate effectively:
  - **FIM-based allocation** — assigns ranks across layers by Fisher Information Matrix (FIM) importance.
  - **Alternating training** — improves robustness under non-IID data and reduces communication.
  - **SVD-based aggregation** — aggregates heterogeneous LoRA updates from clients with different ranks. 

## 🧪 Getting Started

### ⚙️ Environment Setup

**Prerequisites:** [Conda](https://docs.conda.io/); GPU runs need CUDA. See [REPRODUCTION.md](REPRODUCTION.md) for the environment we used.

**Install:**

Estimated times for install: may take one to a few minutes as conda is slow. (TODO replace with uv)

```bash
conda env create --name env.fl --file=setup/environment-final.yml
conda activate env.fl
```

#### HuggingFace Accelerate configuration

Set up the HuggingFace Accelerate configuration:

(Optional) If your system uses multiple GPUs, uncomment the "Multiple GPUs" config options in `setup/accelerate_default_config.yaml`

Copy the accelerate config to the huggingface directory: 

```bash
mkdir -p ~/.cache/huggingface/accelerate/
cp setup/accelerate_default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

**Verify:** Run `bash scripts/experiments/run-cifar100-smoke-test.sh` (this is smoke test: 1 round of training with 2 clients).

### 🚀 Running the Code

#### Rank Estimator

To experiment the Rank Estimator, run the command (estimated to finish in approximately 50 seconds if using the hardware mentioned in [REPRODUCTION.md](REPRODUCTION.md)):
```bash
# Default config: config/rank_estimator_ours.yaml
python scripts/run_rank_estimation.py

# Or with an explicit config path
python scripts/run_rank_estimation.py config/rank_estimator_ours.yaml
```
The results are logged in the terminal upon completion:
```
...
Per client (total rank budget): [288, 432, 576]
Total time to finish the rank estimation task: 50.61s
``` 

See [REPRODUCTION.md](REPRODUCTION.md) for troubleshooting if you run into issues.

#### Rank Utilizer

TODO Youye: add estimated times for run

To utilize the rank budget to fine-tune the model, run one of the scripts in `scripts/experiments/`. For example, our method on CIFAR-100:

```bash
bash scripts/experiments/run-cifar100-Ours.sh
```

Other experiment scripts (CIFAR-100, LEDGAR, IID/non-IID, ablations, baselines) are in `scripts/experiments/`; run any `run-*.sh` from the project root.

**Output:** Logs are written under `log/`, in a path derived from dataset, model, method, and config name (e.g. `log/<dataset>/<model>/<method>/<config_stem>_<timestamp>/`). Each run directory contains `exp_log.txt`.

---

## Codebase overview

### 📁 Project structure

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
│   ├── experiments/   # Experiment run scripts (CIFAR-100, LEDGAR, baselines, ablations)
│   │   └── run-*.sh
│   ├── figures/    # Scripts to generate figures
│   └── run_rank_estimation.py
├── utils/          # Utility functions (including estimator.py for rank estimation)
├── main.py         # Entry point for training
└── test.py         # Evaluation and testing routines
```

For function-level details, see the docstrings in the source code.

---

## Data

See [data/dataset/README.md](data/dataset/README.md) for datasets used in our experiments.

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
