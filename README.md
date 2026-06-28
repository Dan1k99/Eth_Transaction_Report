# Ethereum Transaction Report (Eth_Transaction_Report)

This repository implements a framework for analyzing Ethereum transaction networks, utilizing **Biased Random Walks (TWMDG)** on temporal graphs to generate latent representation embeddings. These embeddings are then evaluated using SVM classifiers for **Temporal Link Prediction**. It includes advanced graph transformations reflecting physical Ethereum Virtual Machine (EVM) telemetry—such as **Gas-Friction bias**, **Router Bipartite Folding**, and **Shadow Nodes** for failed transaction sinks.

---

## 📂 Key Project Files

To keep the repository clean and understandable, here are the core files and their purposes:

### ⚙️ Core Scripts & Workflows
*   **[evaluate.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/evaluate.py)**: The main evaluation pipeline. It constructs the temporal transaction graph, runs the biased random walk embedding generator, and trains/evaluates an SVM classifier for link prediction.
*   **[prepare_dataset.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/prepare_dataset.py)**: An automated utility script that fetches the raw Ethereum transaction zip files, processes them, calculates block-relative gas features, and prepares pickle splits.
*   **[audit.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/audit.py)**: A comprehensive validation and test suite to ensure mathematical correctness of random walk biases, embedding dimensionality, and classifier training.

### 📊 Visualization & Analysis
*   **[visualize.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/visualize.py) & [visualize.html](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/visualize.html)**: A dynamic, interactive web-based visualizer. It hosts a local server showing graphs, allowing users to run step-by-step stochastically simulated walks and observe the effects of gas friction and node failure penalties in real-time.
*   **[plot_graph.py](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/plot_graph.py)**: Generates high-resolution static graphs and plots showcasing walk behavior changes.
*   **[results_explanation.md](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/results_explanation.md)**: A detailed, phase-by-phase analytical breakdown of the A/B testing configurations and the engineering rationale behind their performance.
*   **[ab_test_results.csv](file:///c:/Users/dani9/.gemini/antigravity/scratch/blockchain/ab_test_results.csv)**: Tabular performance statistics (ROC-AUC and Average Precision) across multiple experimental walk setups.

### 📚 Libraries & Datasets
*   **`T-EDGE/`**: The underlying temporal graph representation library and random walk embedding module.
*   **`xblock-network_analysis/`**: The folder structure hosting the underlying transaction edgelists.

---

## 🚀 How to Run the Visualizer

You can run the web visualizer locally:
```bash
python visualize.py
```
This will start a local server (typically at `http://localhost:8000`) and open a browser window displaying the interactive interface. You can load datasets, configure walk parameters, and trigger interactive walks directly.

## 🧪 How to Run the Evaluation & Audit

To verify the installation and run the pipeline checks:
```bash
python audit.py
python evaluate.py
```
