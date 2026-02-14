
#Hardware

Hardware we used:

NVIDIA H100 80GB HBM3 (Driver version 580.65.06, CUDA version 13.0)


# Troubleshooting
## Rank Estimator

If you see the error `torch.cuda.OutOfMemoryError: CUDA out of memory`, wait a few minutes for memory to be released. If it still doesn't work, use a GPU instance with larger memory. 


# Testing

## Manual Test



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

