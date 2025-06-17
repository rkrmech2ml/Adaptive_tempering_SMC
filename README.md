# IS_SMC.py

## Overview

`IS_SMC.py` implements an Importance Sampling Sequential Monte Carlo (IS-SMC) algorithm for probabilistic inference and Bayesian computation. This script is designed to efficiently sample from complex posterior distributions using a sequence of intermediate distributions.

## Features

- Importance sampling with adaptive resampling
- Sequential Monte Carlo framework
- Configurable number of particles and iterations
- Support for custom likelihood and prior functions

## Requirements

- Python 3.x
- NumPy
- (Optional) Matplotlib for visualization

Install dependencies with:
```bash
pip install numpy matplotlib
```

## Usage

```bash
python IS_SMC.py --config config.yaml
```

### Arguments

- `--config`: Path to a YAML or JSON configuration file specifying model parameters.

### Example Configuration

```yaml
num_particles: 1000
num_iterations: 50
resample_threshold: 0.5
```

## Functions

- `initialize_particles()`: Initializes the particle set.
- `compute_weights()`: Calculates importance weights.
- `resample_particles()`: Performs resampling based on weights.
- `run_smc()`: Main loop for the SMC algorithm.

## Customization

Modify the likelihood and prior functions in the script to suit your specific model.


MIT License

