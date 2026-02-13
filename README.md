# 👋 Federated Learning Resource-Aware Heterogeneous LoRA Allocation

## 🧪 Getting Started

### ⚙️ Environment Setup

Ensure you have [Conda](https://docs.conda.io/) installed. 

Then activate the environment using the provided file:

Default:

```bash
conda env create --name env.fl --file=environment-final.yml
conda activate env.fl
```

### 🚀 Running the Code

First, set up the HuggingFace Accelerate configuration:

(Optional) If your system uses multiple GPUs, uncomment the "Multiple GPUs" config options in `accelerate_default_config.yaml`

Copy the accelerate config to the huggingface directory: 

```bash
mkdir -p ~/.cache/huggingface/accelerate/
cp accelerate_default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

Next, launch the training script for the CIFAR-100 dataset:

To experiment the Rank Estimator, run the command:
```bash
# Default config: config/rank_estimator_ours.yaml
python run_rank_estimation.py

# Or with an explicit config path
python run_rank_estimation.py config/rank_estimator_ours.yaml
```
Using an NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0), the Rank Estimator takes approximately 50 seconds to complete. The results are logged in the terminal upon completion:
```
...
Per client (total rank budget): [288, 432, 576]
Total time to finish the rank estimation task: 50.61s
``` 

To utilize the rank budget to fine-tune the model, run the command:

```bash
bash run-cifar100.sh
```

---

## 📁 Project Structure

```
.
├── algorithms/
│   ├── engine/   # Federated learning coordination logic
│   └── solver/   # Local training procedures
├── config/         # YAML configuration files
├── data/           # Dataset cache directory
├── utils/          # Utility functions
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
