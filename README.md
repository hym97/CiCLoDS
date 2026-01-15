# CiCLoDS: Joint Feature Selection and Clustering for Single-cell Spatial Transcriptomics

**CiCLoDS** (**C**lustering **i**n **C**ritical & **L**ow-**D**imensional **S**ubspace) is an unsupervised framework that couples clustering and feature selection in a single objective subject to a user-defined feature budget $p$. Designed for high-dimensional spatial transcriptomics (ST) data, CiCLoDS produces explicit feature subsets aligned with discovered partitions and converges in minutes on commodity hardware.

## 🚀 Key Features

* **Joint Optimization:** Unlike sequential "select-then-cluster" pipelines (e.g., PCA followed by Leiden), CiCLoDS optimizes clustering and feature selection simultaneously, ensuring selected genes directly support the discovered partitions.
* **Budgeted Feature Selection:** Accepts a user-defined budget (e.g., $p=64$ genes) to return compact, assay-ready gene panels.
* **Spatial Awareness:** Naturally incorporates spatial side information via sine-cosine positional encodings (PE) without altering the core optimization objective.
* **High Performance:**
    * **Accuracy:** Outperforms PCA and geneBasis in Adjusted Rand Index (ARI) on hepatocyte zonation tasks.
    * **Structure Preservation:** Achieves superior neighborhood preservation (kNN-overlap AUC of 0.89) compared to baselines.
    * **Efficiency:** Processes large datasets (e.g., 1.27M cells) in under five minutes on a laptop.
* **Synergistic Initialization:** Can serve as a "warm-start" initialization for probabilistic models like BayesSpace, resolving local minima issues and improving segmentation accuracy.

## 🧠 Methodology

CiCLoDS adapts subspace clustering principles to the scale of spatial transcriptomics. It solves the following optimization problem using block coordinate descent:

$$
\min f(K, S, C) = \sum_{K \in \mathcal{K}} \sum_{i \in K} \sum_{j \in S} (g_{ij} - C_{Kj})^2
$$

Where:
* $K$ is the partition of cells into clusters.
* $S$ is the subset of genes limited to budget size $n$.
* $C_{Kj}$ is the centroid of cluster $K$ for gene $j$.
* The objective minimizes the within-cluster sum of squares using only the selected features.

### Spatial Encoding
For spatial transcriptomics, gene expression matrices are augmented with spatial features ($C_{PE}$) derived from positional encodings before entering the optimization loop:
$$X_{aug} = G + C_{PE}$$

## 📊 Usage
Please read the ```tutorial.ipynb``` to see the example.


