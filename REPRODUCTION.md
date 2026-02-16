
# Hardware

Hardware and OS we used:

* GPU: NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0)
* CPU: 8
* Memory: 32Gi
* OS: Ubuntu 22.04.5 LTS.


# Troubleshooting
## Rank Estimator

If you see the error `torch.cuda.OutOfMemoryError: CUDA out of memory`, wait a few minutes for memory to be released. If it still doesn't work, use a GPU instance with larger memory. 


# Testing

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

To reproduce results in **tables and figures** (Table V - VIII, Fig. 1, 3, 5, 6): run from **project root** with the environment activated; see [README](README.md#-getting-started) and [Rank Utilizer](README.md#rank-utilizer) for setup.

1. **Experiments**  
   Run the experiments that correspond to the paper’s settings. Each run writes logs under `log/<dataset>/<model>/<method>/<config_stem>_<timestamp>/`; final and per-round accuracy are in `exp_log.txt`. From the project root:

   - **Ours (CIFAR-100 IID, non-IID 10, non-IID 20):**
     ```bash
     bash scripts/experiments/run-cifar100-Ours.sh
     ```
   - **comparison / ablations:** run the matching scripts in `scripts/experiments/`, e.g. `run-cifar100-iid.sh`, `run-cifar100-iid-ablation.sh`, `run-cifar100-non-iid-20.sh`, etc.
   
2. **Fig. 1 (Accuracy vs. trainable parameters)**  
   Produces the scatter plot of test accuracy (%) vs. number of trainable parameters for method comparison (FedIT, Straggler, Exclusive, LoKr, FFA-LoRA, LEGEND, Fed-HeLLo, HRALoRA). Data obtained from the experiments are hardcoded in the script; no log paths. Run as-is:

   ```bash
   python scripts/figures/fig-param.py
   ```

   Output: `figures/accuracy_vs_params_custom_labels.pdf`.

3. **Fig. 3 (Layer-wise FIM scores and allocated ranks)**  
   The script reads one `exp_log.txt` from an Ours run with FIM (e.g. alternating-training with warm start). Set `log_path` at the top of `scripts/figures/fig-fim-score.py` to your run’s `exp_log.txt`, then run:

   ```bash
   python scripts/figures/fig-fim-score.py
   ```

   Output (by default): `figures/layer-rank.pdf` (allocated rank for selected layers over rounds). To also produce the subfigures (a) and (b), uncomment the blocks in the script that save `fim_mean_std.pdf` (FIM mean ± std per layer) and `rank_mean_std.pdf` (rank mean ± std per layer). The script also contains commented code for `layer-fim.pdf` (FIM per layer over rounds); uncomment that block if needed.

4. **Fig. 5.**  
   Produces the bar chart of accuracy/parameter ratio for different methods. Use log or data obtained from the experiments in the script. Run as-is:

   ```bash
   python scripts/figures/fig-param-performance-ratio.py
   python scripts/figures/fig-train-diff-method.py
   ```

   Output: `figures/ratio_grouped_histogram.pdf` and `cifar_train_plot-diff-method.pdf`.

5. **Fig. 6**  
   Produces the plot of accuracy vs LoRA rank. Use log or data obtained from the experiments in the script. Run as-is:

   ```bash
   python scripts/figures/fig-rank.py
   python scripts/figures/fig-train.py
   ```

   Output: `figures/cifar_rank_plot.pdf` and `figures/cifar_train_plot.pdf`

Run all commands from the **project root** with the project environment activated (see [README](README.md#-getting-started). Figure scripts that read logs require having run the corresponding experiments first; see the [Rank Utilizer](README.md#rank-utilizer) section in the README for how to run experiments.

Troubleshooting: if you face errors, check if the relevant experiment has been run and update the path hardcoded in the scripts that generate figures. 


## Automatic Unit Test

**Unit tests** (in `tests/`): estimator, FIM FL engine, DepthFL engine. From project root with env active (e.g. `conda activate env.fl`):

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
