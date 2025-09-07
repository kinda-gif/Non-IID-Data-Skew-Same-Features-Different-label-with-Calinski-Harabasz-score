# SFDL Cluster Library

## Overview

SFDL Cluster is a Python library designed to assist in federated learning experiments by providing tools for data clustering and distribution under the "Same Features, Different Label" (SFDL) data skew. This type of non Independent and Identically Distributed (non-IID) data distribution is common in real-world federated scenarios, where clients might have similar feature sets but different label distributions due to local variations or specific tasks. The library leverages KMeans clustering with an optimal `k` selection strategy based on the Calinski-Harabasz score to identify natural groupings within the data, and then distributes these clusters to simulate SFDL skew across different clients.

### Calinski-Harabasz Score

The Calinski-Harabasz score, also known as the Variance Ratio Criterion, is a metric used to evaluate the quality of clustering. It is defined as the ratio of the sum of between-clusters dispersion and within-cluster dispersion for all clusters. A higher Calinski-Harabasz score relates to a model with better defined clusters.

Mathematically, the Calinski-Harabasz score (CH) for a set of data E of size $n_E$ that has been clustered into $k$ clusters is given by:

$CH = \frac{Tr(B_k)}{Tr(W_k)} \times \frac{n_E - k}{k - 1}$

Where:
- $Tr(B_k)$ is the trace of the between-group dispersion matrix.
- $Tr(W_k)$ is the trace of the within-cluster dispersion matrix.
- $n_E$ is the number of data points.
- $k$ is the number of clusters.

In simpler terms, $Tr(B_k)$ measures how spread out the cluster centers are from each other, and $Tr(W_k)$ measures how compact the clusters are internally. A good clustering will have clusters that are far apart from each other (high $Tr(B_k)$) and internally compact (low $Tr(W_k)$). The Calinski-Harabasz score effectively captures this balance, making it a valuable tool for determining the optimal number of clusters in a dataset.

### Mathematical Definition of SFDL Skew

Same features, different label skew in non-IID data refers to the case where the conditional distribution of labels (P(Y∣X)) varies across different subsets of the data, even though the feature distributions (P(X)) remain consistent. In this scenario, the way labels (Y) are associated with a given feature set (X) is not uniform across subsets.

Mathematically, for any subsets i and j, this can be expressed as:

$P_i (Y|X) \neq P_j (Y|X) \quad \text{for } i \neq j$

This implies that, while the features themselves are similarly distributed across the subsets, the labels corresponding to a particular feature set differ, potentially due to variations in local labeling conventions, sensor biases, or demographic differences in the subsets.

## Features

- **Optimal K-Means Clustering**: Automatically determines the optimal number of clusters (`k`) using the Calinski-Harabasz score, ensuring efficient and meaningful data partitioning.
- **SFDL Data Distribution**: Implements a specialized data distribution strategy to create datasets with "Same Features, Different Label" skew, crucial for realistic federated learning simulations.
- **Federated Learning Compatibility**: Generates data splits suitable for simulating diverse client environments in federated learning setups.
- **Easy Integration**: Provides straightforward functions for data clustering and client-specific data distribution.
- **Pythonic Interface**: Offers a clean and intuitive API for seamless integration into existing machine learning workflows.

## Installation

To install SFDL Cluster, you can use pip:

```bash
pip install sfdl-cluster
```

Alternatively, if you have cloned the repository, you can install it from the local directory:

```bash
pip install .
```

## Usage

Here's a basic example of how to use SFDL Cluster to partition your data:

```python
import pandas as pd
from sfdl_cluster.cluster import find_optimal_k, cluster_and_distribute_sfdl

# Example: Create a dummy dataset
data = {
    'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature_2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Find optimal k (number of clusters)
# optimal_k = find_optimal_k(df[['feature_1', 'feature_2']])
# print(f"Optimal k: {optimal_k}")

# Distribute data to clients with SFDL skew
# client_datasets = cluster_and_distribute_sfdl(df, features=['feature_1', 'feature_2'], num_clients=2)

# For demonstration, let's assume optimal_k is 2 and num_clients is 2
client_datasets = cluster_and_distribute_sfdl(df, features=['feature_1', 'feature_2'], num_clients=2, optimal_k=2)

for client_id, client_df in client_datasets.items():
    print(f"\nClient {client_id} data shape: {client_df.shape}")
    print(client_df.head())
```

## API Reference

### `find_optimal_k(data, max_k=10)`

This function determines the optimal number of clusters (`k`) for KMeans clustering using the Calinski-Harabasz score.

- **`data`** (pandas.DataFrame or numpy.ndarray): The input data for clustering. This should contain only the features to be used for clustering.
- **`max_k`** (int, optional): The maximum number of clusters to consider when searching for the optimal `k`. **Default value is `10`**.

**Returns**:
- **`int`**: The optimal number of clusters (`k`).

### `cluster_and_distribute_sfdl(df, features, num_clients, optimal_k=None)`

This function clusters the data and distributes it to clients to simulate "Same Features, Different Label" (SFDL) skew.

- **`df`** (pandas.DataFrame): The input DataFrame containing your dataset, including features and labels.
- **`features`** (list of str): A list of column names in `df` that represent the features to be used for clustering.
- **`num_clients`** (int): The desired number of clients to distribute the data among.
- **`optimal_k`** (int, optional): The number of clusters to use for KMeans. If `None`, `find_optimal_k` will be called internally to determine it. **Default value is `None`**.

**Returns**:
- **`dict`**: A dictionary where keys are client IDs (integers) and values are pandas.DataFrames, each representing the data assigned to a specific client.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Note: A `LICENSE` file is not included in the provided zip, this is a placeholder.)
