
#Hardware

Hardware we used:

NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0)


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

1. **Per-client rank budgets (paper table)**  
   Run rank estimation for the heterogeneous client groups (e.g. 2GB / 4GB / 8GB GPU, different network speeds). This prints the per-client total rank budget list used in the paper.

   ```bash
   python scripts/run_rank_estimation.py config/rank_estimator_ours.yaml
   ```

   Example output: `Per client (total rank budget): [288, 432, 576]`.

2. **Figure: Rank budget vs. GPU memory and vs. network speed (Fig. 4)**  
   Generates the dual-axis plot (rank vs. memory in blue, rank vs. network speed in green). Output is saved under the project `figures/` directory.

   ```bash
   python scripts/figures/fig_rank_budget_memory_and_network.py
   ```

   Output file: `figures/rank_vs_memory_and_network_speed_combined.pdf`.

3. **Table: Memory breakdown comparison (estimated vs. profiled)**  
   Runs memory profiling and writes a LaTeX table comparing estimated and profiled memory for LoRA (query/value, 2GB GPU). Requires a GPU.

   ```bash
   python scripts/figures/fig_memory_breakdown_comparison.py qv
   ```

   LaTeX table is written to `results/diagrams/memory_breakdown_comparison_lora_qv.tex`.

Activate the project environment (e.g. `conda activate env.fl`) and run these commands from the **project root**. If the rank estimator hits GPU OOM, see the “Rank Estimator” note under [Troubleshooting](#rank-estimator) above.

### Rank Utilizer

TODO review, test, and refine the content below

To reproduce **tables and figures** in the paper about rank allocation (FIM-based) and aggregation (SVD, weighted average):

1. **Result tables (accuracy)**  
   Run the experiments that correspond to the paper’s settings. Each run writes logs under `log/<dataset>/<model>/<method>/<config_stem>_<timestamp>/`; final and per-round accuracy are in `exp_log.txt`. From the project root:

   - **Ours (CIFAR-100 IID, non-IID 10, non-IID 20):**
     ```bash
     bash scripts/experiments/run-cifar100-Ours.sh
     ```
   - **Baselines / ablations:** run the matching scripts in `scripts/experiments/`, e.g. `run-cifar100-FedHello.sh`, `run-cifar100-FlexLoRA.sh`, `run-cifar100-HetLoRA.sh`, `run-cifar100-iid.sh`, `run-cifar100-non-iid-10.sh`, `run-cifar100-non-iid-20.sh`, etc.

   To build the paper’s accuracy tables, point to the desired run directories and parse `exp_log.txt` (e.g. final accuracy or round-wise accuracy).

2. **Figure: Training curves (accuracy vs round)**  
   Produces the plot of accuracy over communication rounds (e.g. IID, non-IID 10, non-IID 20). The script reads three `exp_log.txt` paths that are **hardcoded** in the file. After running the three Ours runs above, edit the three `Path(...)` paths at the top of the script to your `log/.../exp_log.txt` paths, then run:

   ```bash
   python scripts/figures/fig-train.py
   ```

   Output: `figures/cifar_train_plot.pdf`.

3. **Figure: Rank vs accuracy**  
   Produces the plot of accuracy vs LoRA rank. Uses data hardcoded in the script (no log paths). Run as-is:

   ```bash
   python scripts/figures/fig-rank.py
   ```

   Output: `figures/cifar_rank_plot.pdf`.

4. **Figure: FIM-based rank allocation (allocated rank per layer over rounds)**  
   Produces the plot of allocated rank for selected layers over training rounds. The script reads one `exp_log.txt` that contains `fim score` and `rank list` lines (from an Ours run with FIM, e.g. alternating-training with warm start). Set `log_path` at the top of the script to your run’s `exp_log.txt`, then run:

   ```bash
   python scripts/figures/fig-fim-score.py
   ```

   Output: `figures/layer-rank.pdf`.

5. **Figure: Parameter–performance ratio (method comparison)**  
   Produces the bar chart of accuracy/parameter ratio for different methods. Uses data hardcoded in the script. Run as-is:

   ```bash
   python scripts/figures/fig-param-performance-ratio.py
   ```

   Output: `figures/ratio_grouped_histogram.pdf`.

6. **Figure: Fig. 1**
   ```bash
   python scripts/figures/fig-param.py
   
   ```
   TODO

Run all commands from the **project root** with the project environment activated (e.g. `conda activate env.fl`). Figure scripts that read logs require having run the corresponding experiments first; see the [Rank Utilizer](README.md#rank-utilizer) section in the README for how to run experiments.



