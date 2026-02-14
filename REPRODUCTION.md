
#Hardware

Hardware and OS we used:

GPU: NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0)
CPU: 8
Memory: 32Gi
OS: Ubuntu 22.04.5 LTS.


# Troubleshooting
## Rank Estimator

If you see the error `torch.cuda.OutOfMemoryError: CUDA out of memory`, wait a few minutes for memory to be released. If it still doesn't work, use a GPU instance with larger memory. 


# Testing

## Automatic Unit Test

**Unit tests** (in `tests/`): estimator, FIM FL engine, DepthFL engine. No GPU needed; mocks used where needed. From project root with env active (e.g. `conda activate env.fl`):

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Manual Test

### Rank Estimator

To reproduce the **tables and figures** in the paper related to the rank estimator:

Activate the project environment (see [README](README.md#-getting-started) for setup) and run these commands from the **project root**. If the rank estimator hits GPU OOM, see the “Rank Estimator” note under [Troubleshooting](#rank-estimator) above.

1. **Table III**  
   Run rank estimation for heterogeneous groups (e.g. 2GB/4GB/8GB GPU, different network speeds); prints the per-client rank budget list used in the paper.

   ```bash
   python scripts/run_rank_estimation.py config/rank_estimator_ours.yaml
   ```

   Example output: `Per client (total rank budget): [288, 432, 576]`.

   We used the output above and manually created Table III in the paper. 

2. **Fig. 4**  
   Generates the dual-axis plot (rank vs. memory in blue, rank vs. network speed in green).

   ```bash
   python scripts/figures/fig_rank_budget_memory_and_network.py
   ```

   Output file: `figures/rank_vs_memory_and_network_speed_combined.pdf`.

   Due to the inherent randomness in linear regression, the output may be slightly different from the numbers in our paper. 

3. **Table IV: MEMORY BREAKDOWN COMPARISON TABLE**  
   Runs memory profiling and writes a LaTeX table comparing estimated and profiled memory for LoRA (query/value, 2GB GPU). Requires a GPU.

   ```bash
   python scripts/figures/fig_memory_breakdown_comparison.py qv
   ```

   LaTeX table is written to `results/diagrams/memory_breakdown_comparison_lora_qv.tex`.

   Due to the inherent randomness in linear regression, the output may be slightly different from the numbers in our paper. 

### Rank Utilizer

To reproduce results in **tables and figures** (Table V - VIII, Fig. 1, 3, 5, 6): run from **project root** with the environment activated; see [README](README.md#-getting-started) and [Rank Utilizer](README.md#rank-utilizer).

1. **Experiments**  
   Run the experiments that correspond to the paper’s settings. Each run writes logs under `log/<dataset>/<model>/<method>/<config_stem>_<timestamp>/`; final and per-round accuracy are in `exp_log.txt`. From the project root:

   - **Ours (CIFAR-100 IID, non-IID 10, non-IID 20):**
     ```bash
     bash scripts/experiments/run-cifar100-Ours.sh
     ```
   - **Baselines / ablations:** run the matching scripts in `scripts/experiments/`, e.g. `run-cifar100-FedHello.sh`, `run-cifar100-FlexLoRA.sh`, `run-cifar100-HetLoRA.sh`, `run-cifar100-iid.sh`, `run-cifar100-non-iid-10.sh`, `run-cifar100-non-iid-20.sh`, etc.
   





