# 👋 Heterogeneous-Resource-Aware Federated Learning with Intelligent LoRA Allocation and Aggregation

## 🧪 Getting Started

### ⚙️ Environment Setup

Ensure you have [Conda](https://docs.conda.io/) installed. 

Check out the branch ccgrid2026-v2

Then update `prefix: /home/yourhomepath/anaconda3/envs/env.fl` in the `environment-final.yml` file based on your path. 

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
cp accelerate_default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

Next, launch the training script for the CIFAR-100 dataset:

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
├── log/            # Output logs and saved results
├── model/          # Model definitions
├── utils/          # Utility functions
├── main.py         # Entry point for training
└── test.py         # Evaluation and testing routines
```
---

## 📄 Citation

If you find this work useful for your research, please cite papers
```
@inproceedings{xie2026hralora,
  title={Heterogeneous-Resource-Aware Federated Learning with Intelligent LoRA Allocation and Aggregation},
  author={Xie, Youye and Lian, Yao and Chen, Kevin and Latif, Abdul and Zhao, Lingzhi and Farivar, Reza},
  booktitle={Proceedings of the 26th IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
  year={2026},
  organization={IEEE/ACM}
}
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
