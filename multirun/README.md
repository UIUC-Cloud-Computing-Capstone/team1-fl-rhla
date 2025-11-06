## Overview

This directory contains the code needed to schedule multiple runs of `run-cifar100.sh` with varying configurations.

## Run experiment

1. Add experimental configurations to `./multirun/experiments.json`
2. Run `python ./multirun/run_experiments.py`

## Parse result
1. Run `python ./multirun/parse_result.py`
3. Test accuracy result is saved in `./multirun/results`
