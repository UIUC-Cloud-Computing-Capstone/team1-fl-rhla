
#Hardware

Hardware we used:

NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0)


# Troubleshooting
## Rank Estimator

If you see the error `torch.cuda.OutOfMemoryError: CUDA out of memory`, wait a few minutes for memory to be released. If it still doesn't work, use a GPU instance with larger memory. 


# Testing

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

TODO

## Unit Test

Unit tests for the rank estimator (`utils/estimator.py`) live in `tests/test_estimator.py`. Activate the project environment first (e.g. `conda activate env.fl`), then from the project root run:

```bash
python -m unittest tests.test_estimator -v
```

Or run the test file directly:

```bash
python tests/test_estimator.py -v
```

The `-v` flag enables verbose output. These tests do not require a GPU; they use mocks where needed.

