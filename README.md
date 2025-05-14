# Feature-Based Network Construction (FBNC)

This repository implements the **Feature-Based Network Construction (FBNC)** framework, a gradient-based algorithm to construct and modify weighted directed graphs that **exactly match a set of prescribed structural features**.

The method supports both:
- **Hard constraint sampling**, generating networks that satisfy features exactly.
- **What-if analysis**, modifying an initial network to desire a feature while minimally while adjusting the network.

FBNC is introduced in the paper:

> **Franssen, C., Berkhout, J., & Heidergott, B. (2024).**  
> *Feature-Based Network Construction: From Sampling to What-if Analysis*.  
> [arXiv:2412.05124](https://arxiv.org/abs/2412.05124)

---

## 📦 Prerequisites

- Python 3.12 (tested on 3.12.2)
- Conda (recommended for environment setup)

Set up the environment:

```bash
conda env create -f environment.yml
conda activate fbnc
```

---

## ⚙️ Configuration

All configurable parameters are defined near the top of `main.py` under the **CONFIGURATION** block.

### 🎯 Problem Specification

Problems can be specified in `main.py`. Below setting reproduces FIG. 16 (left) in the paper. Other figures involving the financial network example can be reproduced using `data/financial_network_example.csv`.

```python
# FBNC problem type
FBNC_TYPE       = "what-if" # 'sampling' or 'what-if'
FEATURES        = [kemeny_constant(31.638)] # List of feature(target) to be used
NORM            = 2 # Choose L1 or L2 norm
LB              = 0.0 # Lower bound for edge weights
UB              = 1.0 # Upper bound for edge weights
N_SAMPLES       = 2 # Number of samples to be generated (only for sampling mode)

# Network settings
W0              = load_csv_matrix(DATA_DIR / "social_network_example.csv") # Initial weights for what-if analysis
N,_             = W0.shape
PARAMS          = np.ones((N,N)) - np.eye(N) # Which edges may be used?
UNDIRECTED_GRAPH  = False
MARKOVIAN_GRAPH   = True
```

Note: any features used should be specified in `network/features.py`. Common features include:

- Node strengths (in/out)
- Triangular closure
- Modularity
- Assortativity
- Stationary distribution
- Kemeny constant
- Effective resistance

### ⚙️ Optimization Settings

Located in `main.py`, e.g.:

```python
FBNC_ALG_CONFIG = SimpleNamespace(
    max_iter    = 10000, # Number of iterations (T)
    beta        = 0.5, # Armijo rescaling parameter
    sigma       = 0.5, # Armijo strictness parameter
    alpha_ini   = 1e-3,  # Armijo step size parameter
    omega       = 1e-3, # Convergence threshold
)
```

---

## 🚀 Running FBNC

Run a sampling procedure or what-if analysis via:

```bash
python main.py
```

The script:
- Initializes the graph and features
- Runs FBNC with steepest feasible descent
- Outputs results to `results/`

---

## 📊 Output & Results

Each run creates a subdirectory in `results/` where in sampling mode:

```
results/run_YYYYMMDD_HHMMSS/
├── network_001.npz         # Sampled network 1
├── network_002.npz         # Sampled network 2
├── ...
```
and in what-if mode:
```
results/run_YYYYMMDD_HHMMSS/
├── network.npz         # Fitted network
```

---

## 🔖 Citation

If you use this repository, please cite the original paper:

```
@article{franssen2024fbnc,
  title={Feature-Based Network Construction: From Sampling to What-if Analysis},
  author={Franssen, Christian and Berkhout, Joost and Heidergott, Bernd},
  journal={arXiv preprint arXiv:2412.05124},
  year={2024}
}
```   

---

## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 🤝 Contact

Feel free to open an issue or reach out with questions at c.p.c.franssen [at] vu.nl.
