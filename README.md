# GPU Scheduler Fairness Analysis

Empirical and theoretical analysis of GPU scheduling fairness under memory constraints.

## Project Overview

This project investigates if and how memory-aware scheduling can improve fairness when scheduling GPU compute workloads. The project combines:

1. **Theoretical Simulator**: Discrete-event simulation of GPU schedulers (FIFO, Fair, Memory-Aware)
2. **Empirical Measurements**: Real GPU workload measurements using PyTorch

## Project Structure

```
├── src/
│   ├── scheduler.py          # Base scheduler class and implementations (FIFO, Fair, Memory-Aware)
│   ├── workload.py            # Job/workload definitions
│   ├── memory_manager.py      # GPU memory tracking and swap management
implementation
├── simulator.py               # Simulator demo and comparison
├── measurement.py            # Empirical GPU fairness measurement script
├── results/                  # Measurement outputs from measurement.py
│   ├── results_*.csv         # CSV files with detailed metrics (4 scenarios × 12 trials)
│   ├── fairness_comparison_*.png  # Fairness visualization
│   └── results.txt          # Output of measurement.py
└── README.md
```

## Installation

```bash
pip install torch scipy pandas numpy matplotlib
```

**Requirements:**

- Python 3.7+
- CUDA-enabled GPU (for measurement.py)
- Tested on Google Colab Tesla T4

## Usage

### Simulator (Theoretical Analysis)

```bash
# Run scheduler comparison demo
python simulator.py
```

This runs three scheduling policies on a sample workload:

- **FIFO**: First-in-first-out scheduling
- **Fair Scheduler**: Virtual runtime-based fair scheduling (inspired by CFS)
- **Memory-Aware**: Considers both fairness and memory residency to minimize swaps

### Measurements (Empirical Analysis)

```bash
# Run GPU fairness measurements
python measurement.py
```

This measures actual GPU scheduler behavior using PyTorch workloads, comparing:

- **Sequential execution** (batch arrival, FIFO scheduling)
- **Concurrent execution** (staggered arrival, realistic workload)
- **Homogeneous workloads** (equal memory/duration)
- **Heterogeneous workloads** (mixed small/large jobs)

**Google Colab:**

```python
!pip install scipy
!python measurement.py
```

## Output

### Measurement Results

**CSV Files** (4 scenarios × 12 trials each):

- `results_homogeneous_sequential_*.csv`
- `results_heterogeneous_sequential_*.csv`
- `results_homogeneous_concurrent_*.csv`
- `results_heterogeneous_concurrent_*.csv`

**Visualization:**

- `fairness_comparison_*.png` - Bar chart comparing Jain's Fairness Index across scenarios
- `gpu_measurements_*.png` - Turnaround time box plots

**Console Output:**

- Fairness metrics (Jain's Index)
- T-test results (sequential vs concurrent)
- Small_1 vs Small_2 analysis
- Wait time comparisons
