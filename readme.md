## TailoredBench

This project implements the TAILOREDBENCH method proposed in the paper "Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation" **(ACL 2025 MAIN)** for efficient estimation of target model performance on complete benchmark datasets under limited inference budgets. The core idea is to construct globally representative core sets adapted to source models, adaptively select local source models based on prediction consistency between target and source models, and combine scalable K-Medoids clustering with calibrated estimation strategies to achieve customized performance prediction.

### Project Structure

```bash
Tailoredbench /
├── main.py                   # Main entry point with argument parsing and experiment scheduling
├── config/                   # Global configuration
│   ├── __init__.py
│   └── settings.py           # Clustering, distance functions, path configurations, etc.
├── data/                     # Data loading and processing
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading and validation
│   └── data_processor.py     # Model accuracy computation and metric conversion
├── models/                   # Clustering and prediction model modules
│   ├── __init__.py
│   ├── clustering.py         # G-set construction, CLARA clustering implementation
│   └── prediction.py         # Parallel K-Medoids prediction process
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── metrics.py            # Correlation and error metric calculation functions
│   ├── file_utils.py         # Excel file construction and writing
│   └── parallel_utils.py     # Multiprocessing scheduler and worker wrapper
├── experiments/              # Core experiment logic
│   ├── __init__.py
│   └── score_cc_set.py       # Main evaluation pipeline: G-set construction, N-set clustering, performance estimation
├── BenchmarkDatasets/        # Raw input data directory for each dataset
│   ├── arc_challenge/        # Each subdirectory contains required `.npy` files for corresponding dataset
│   ├── hellaswag/
│   ├── pope/
│   └── ...
├── AnchorPoints/             # Experiment results output directory
├── scikit-learn-extra/       # Modified K-Medoids implementation for efficient clustering under fixed core set conditions
└── requirements.txt
```

------

###  Core Functionality

1. **main.py**
    Provides command-line argument interface for setting experimental parameters such as dataset, number of source models, cluster count, evaluation metric types, parallel processes, etc., and launches the main experimental pipeline`score_cc_set_experiment()`.
2. **score_cc_set_experiment()**
    `experiments/score_cc_set.py`as the main experiment entry point. Sequentially executes data loading, G-set construction, local source model selection, N-set construction, performance estimation, and evaluation metric recording.
3. **score_cc_set_selection()**（clustering.py）
    Based on source model prediction accuracy, uses CLARA algorithm for standardization and Manhattan distance clustering to select globally representative samples as G-set.
4. **gpu_k_medoids_parallel()**（prediction.py）
    Performs GPU-parallel K-Medoids clustering and prediction scoring for each batch of target models, executing scaling calibration based on consistency between target and local source model performance.
5. **compute_correlations()**（metrics.py）
    Computes various metrics (Kendall's tau, Pearson, MAE, MSE) to evaluate correlation and error between estimated performance and ground truth labels.
6. **create_excel_with_sheets() & append_to_excel()**（file_utils.py）
    Stores various metrics in Excel format during experiments, separately storing estimated value means and errors.

### 🛠️ Custom K-Medoids Algorithm (scikit-learn-extra Modifications)

The project uses the `CLARA` clustering algorithm from the `scikit-learn-extra` library as the core K-Medoids implementation. To meet the specific constraint requirements of fixed core points (i.e., G-set fixed as initial centroids), the library has been customized in the following files:

- `scikit-learn-extra/sklearn_extra/cluster/_k_medoids.py`
- `scikit-learn-extra/sklearn_extra/cluster/_k_medoids_helper.pyx`

This modified version provides:

- Ability to fix initial center points
- Support for flexible clustering based on various distance metrics (such as Manhattan distance)

### Installation

Navigate to the `scikit-learn-extra/` directory for installation:

```bash
cd scikit-learn-extra
pip install -e .
```

Or set `PYTHONPATH` to specify the path for local package usage:

```bash
export PYTHONPATH=$PYTHONPATH:/absolute/path/to/Tailoredbench/scikit-learn-extra
```

### 🧪 Usage

#### 1.Environment Setup

The project requires the following main components：

- Python ≥ 3.10
- numpy
- scipy
- scikit-learn
- sklearn-extra
- joblib
- openpyxl
- tqdm
- torch

Environment initialization is recommended using：

```bash
pip install -r requirements.txt
```

#### 2.  Example Usage

Run directly through main.py. Parameters can be adjusted in main.py and config/settings.py:

```bash
python main.py \
    --dataset all \
    --results_dir ./results \
    --point_counts 20 \
    --num_cc_set 10 \
    --distance manhattan \
    --num_runs 100 \
    --num_processes 24
```

#### 3. Main Parameters

| Parameter                     | Description                                                   |
| ----------------------------- | ----------------------------------------------------------    |
| `--dataset`                   | Dataset name to evaluate (e.g., gsm8k, Hellaswag)             |
| `--results_dir`               | Experiment results output directory                           |
| `--point_counts`              | Number of N-set size **excluding** G-set size **(!!! Very Very Important !!!)**<span> &nbsp; &nbsp; &nbsp; </span>**point_counts + num_cc_set(G-set) = N-set**|
| `--num_cc_set`                | Number of K-Medoids clusters when constructing G-set          |
| `--distance`                  | Distance metric used for clustering (default: `manhattan`)    |
| `--num_runs`                  | Number of repeated runs per setting                           |
| `--num_processes`             | Number of parallel processes (recommended: CPU core count)    |

------

### 📊 Output Results

- Experiment results will be saved in txt files under the path `./AnchorPoints/{dataset_name}_result/clusterunionset_quantile/`
- The summary file `dynamic_cpu_ccset:{args.num_cc_set}_cset:{args.point_counts}_time:{formatted_datetime}_distance:{args.distance}_randomseed_clara_union.txt` contains five sections: `pearsonr`, `kendalltau`, `spearmanr`, `MAE`, `MSE`, recording means and standard errors of estimated values respectively

------

### 📎Reference Paper

Yuan P., Zhang Y., Feng S., et al. Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation[C]//arXiv preprint arXiv:2502.13576, 2025.