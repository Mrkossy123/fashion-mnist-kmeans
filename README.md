# Fashion-MNIST Clustering from Scratch with K-Means

Implementation of the **K-means clustering** part of a university Machine Learning assignment on the **Fashion-MNIST** dataset.

The assignment asks for unsupervised grouping of clothing images into **K = 10 clusters** using different data representations and distance measures. In this repository, the implemented and fully working part is the **K-means algorithm from scratch**, together with dataset preprocessing, feature construction, and clustering evaluation. The assignment specification requires use of the **Fashion-MNIST training set**, support for **two image representations (R1 and R2)**, and evaluation using **Purity** and **F-measure**. This repository currently covers that K-means part directly. 

---

## Table of Contents

- [Overview](#overview)
- [Assignment Goal](#assignment-goal)
- [Dataset](#dataset)
- [Implemented Scope](#implemented-scope)
- [Data Representations](#data-representations)
- [Distance Metrics](#distance-metrics)
- [Evaluation Metrics](#evaluation-metrics)
- [Algorithm Description](#algorithm-description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Command-Line Arguments](#command-line-arguments)
- [Example Executions](#example-executions)
- [Example Output](#example-output)
- [Interpretation of Results](#interpretation-of-results)
- [Implementation Notes](#implementation-notes)
- [Limitations](#limitations)
- [Possible Extensions](#possible-extensions)
- [Academic Context](#academic-context)
- [Author](#author)

---

## Overview

This project performs **unsupervised clustering** on grayscale clothing images from the Fashion-MNIST dataset.

The main idea is to group visually similar images into clusters **without using the labels during training**, and then evaluate the quality of the resulting clusters using the true labels only at the end.

The implementation supports:

- **R1 representation**: raw normalized pixel vector of length `784`
- **R2 representation**: normalized brightness histogram with configurable number of bins
- **Distance metrics**:
  - Euclidean (`L2`)
  - Manhattan (`L1`)
  - Cosine distance
- **Evaluation metrics**:
  - Purity
  - F-measure (mean F1 across clusters)

The entire clustering logic is implemented **from scratch**, without using a ready-made K-means implementation from libraries such as scikit-learn.

---

## Assignment Goal

The goal of the assignment is to analyze the structure of the Fashion-MNIST image set through **clustering**, by partitioning the data into **K = 10 groups**. The specification explicitly refers to two feature representations:

- **R1**: image flattened into a vector of `28 x 28 = 784` values
- **R2**: brightness histogram using `M` bins, typically `16`, `32`, `64`, or `128`

For the K-means part, the assignment requires experimentation with:

- **Euclidean distance (L2)**
- **Manhattan distance (L1)**
- **Cosine distance**

The assignment also requires evaluation with:

- **Purity**
- **F-measure**

---

## Dataset

The dataset used is **Fashion-MNIST**, a standard benchmark dataset of grayscale images of fashion products.

### Image properties

- Image size: `28 x 28`
- Channels: grayscale
- Pixel range: `0-255`
- Number of classes: `10`

### Classes

| Label | Class Name |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Only the **training set** is used for clustering.

To keep execution practical for a from-scratch implementation, the code also supports selecting a **balanced subset** of the training set, meaning the same number of samples is selected from each class.

---

## Implemented Scope

This repository currently implements the following:

- Loading the **Fashion-MNIST training set only**
- Optional selection of a **balanced subset** per class
- Construction of **R1** and **R2** feature representations
- K-means clustering **from scratch**
- Support for **L2**, **L1**, and **cosine** distances
- Computation of:
  - final cluster assignments
  - cluster centers
  - inertia
  - Purity
  - F-measure
- Detailed reporting for each cluster:
  - size
  - majority class
  - label distribution
  - TP / FP / FN / F1

### Current scope clarification

This repository **does not yet include**:

- K-medoids with KL-based distance
- Hierarchical clustering
- visualization plots
- automated report export

Those parts are mentioned in the assignment as broader requirements, but the current codebase focuses specifically on the **K-means implementation**.

---

## Data Representations

### R1 — Raw Pixel Vector

Each image is:

1. converted to `float32`
2. normalized by dividing by `255.0`
3. flattened from `28 x 28` to a vector of length `784`

This representation preserves the full pixel information of the image.

### R2 — Brightness Histogram

Each image is transformed into a histogram of pixel intensities.

- The number of bins is configurable using `--bins`
- Typical values: `16`, `32`, `64`, `128`
- The histogram is normalized so that the sum of all bin values equals `1`

This representation treats the image as an intensity distribution rather than a spatial grid of pixels.

---

## Distance Metrics

The implementation supports three distance metrics.

### 1. Euclidean Distance (`l2`)

Measures straight-line distance between vectors.

Used for standard K-means and often provides a good baseline.

### 2. Manhattan Distance (`l1`)

Measures the sum of absolute differences between vector components.

Can behave differently from Euclidean distance, especially when the features have sparse or structured variation.

### 3. Cosine Distance (`cosine`)

Measures angular dissimilarity between vectors.

Useful when the magnitude of vectors matters less than their direction.

---

## Evaluation Metrics

After clustering is completed, the true labels are used **only for evaluation**.

### Purity

For each cluster, the cluster is assigned the label of the **majority class** among its members.

Purity is then computed as:

- the number of correctly matched samples,
- divided by the total number of samples.

Higher purity means cleaner and more homogeneous clusters.

### F-measure

For each cluster:

- **TP**: samples in the cluster that belong to the majority class
- **FP**: samples in the cluster that belong to other classes
- **FN**: samples outside the cluster that belong to the majority class

Then the cluster F1-score is computed as:

```text
F1 = 2TP / (2TP + FP + FN)
```

The final reported value is the **mean F1-score across all clusters**.

---

## Algorithm Description

The implemented K-means algorithm follows the standard iterative procedure:

1. **Initialize** `K` cluster centers by randomly selecting samples from the dataset
2. **Assign** each sample to the nearest center based on the chosen metric
3. **Recompute** each center as the mean of all samples assigned to that cluster
4. Repeat until convergence

### Convergence criteria

The algorithm stops when either:

- assignments stop changing, or
- the center movement becomes smaller than a given tolerance (`tol`)

### Empty cluster handling

If a cluster becomes empty during an iteration, the implementation keeps the **previous center** for that cluster.

### Inertia

The code also prints the final **inertia**, which is the sum of squared distances of points to their assigned center.

---

## Project Structure

```text
.
├── kmeans_completed.py   # Main implementation of the clustering pipeline
└── README.md            # Project documentation
```

---

## Requirements

- Python `3.9+`
- NumPy
- TensorFlow or Keras

The dataset is loaded using `tensorflow.keras.datasets.fashion_mnist` (or `keras.datasets.fashion_mnist` as fallback).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repository-url>
cd <repository-folder>
pip install numpy tensorflow
```

If you already have a working Keras/TensorFlow environment, that is enough.

---

## How to Run

### Default execution

```bash
python kmeans_completed.py
```

Default configuration:

- `representation = R1`
- `distance = l2`
- `k = 10`
- `samples-per-class = 100`
- `max-iter = 50`
- `tol = 1e-6`
- `seed = 42`

This means the default run uses a balanced subset of **1000 images total** (`100 samples x 10 classes`).

---

## Command-Line Arguments

| Argument | Description | Default |
|---|---|---:|
| `--representation` | Data representation: `R1` or `R2` | `R1` |
| `--distance` | Distance metric: `l2`, `l1`, `cosine` | `l2` |
| `--bins` | Number of histogram bins for `R2` | `32` |
| `--k` | Number of clusters | `10` |
| `--samples-per-class` | Number of samples per class for a balanced subset. Use `0` or a negative value for the full training set | `100` |
| `--max-iter` | Maximum number of K-means iterations | `50` |
| `--tol` | Convergence tolerance | `1e-6` |
| `--batch-size` | Batch size used for distance computation | `512` |
| `--seed` | Random seed | `42` |
| `--quiet` | Suppress per-iteration logs | `False` |

---

## Example Executions

### 1. Default run

```bash
python kmeans_completed.py
```

### 2. R1 with Euclidean distance

```bash
python kmeans_completed.py --representation R1 --distance l2
```

### 3. R1 with Manhattan distance

```bash
python kmeans_completed.py --representation R1 --distance l1
```

### 4. R1 with cosine distance

```bash
python kmeans_completed.py --representation R1 --distance cosine
```

### 5. R2 with 16-bin histogram

```bash
python kmeans_completed.py --representation R2 --distance l2 --bins 16
```

### 6. R2 with 32-bin histogram and Manhattan distance

```bash
python kmeans_completed.py --representation R2 --distance l1 --bins 32
```

### 7. R2 with 64-bin histogram and cosine distance

```bash
python kmeans_completed.py --representation R2 --distance cosine --bins 64
```

### 8. Larger balanced subset

```bash
python kmeans_completed.py --representation R1 --distance l2 --samples-per-class 300
```

### 9. Full training set

```bash
python kmeans_completed.py --samples-per-class 0
```

> Running on the full training set is significantly slower.

### 10. Silent execution without iteration logs

```bash
python kmeans_completed.py --quiet
```

---

## Example Output

A sample run with:

```bash
python kmeans_completed.py
```

produced the following summary on a balanced subset of `1000` images:

```text
Representation: R1
Distance      : l2
K             : 10
Τελικό σχήμα χαρακτηριστικών: (1000, 784)
...
Σύγκλιση σε 25 επαναλήψεις
Τελικό inertia: 32296.939453

Purity     : 0.552000
F-measure  : 0.558198
```

This indicates that the implementation runs successfully, converges normally, and produces meaningful clustering statistics.

---

## Interpretation of Results

On Fashion-MNIST, some classes are easier to cluster than others.

Classes such as:

- **Trouser**
- **Sneaker**
- **Ankle boot**
- **Bag**

usually form cleaner clusters because their visual structure is more distinctive.

Classes such as:

- **T-shirt/top**
- **Shirt**
- **Pullover**
- **Coat**
- **Dress**

often overlap more in pixel space, especially under the simple `R1` representation.

Therefore, moderate purity and F-measure values are expected for a from-scratch K-means baseline.

---

## Implementation Notes

A few important implementation details:

- The code uses **only the training set**, as required by the assignment
- Labels are **not used during clustering**, only during evaluation
- R2 histograms are normalized so they behave as distributions
- Distance computation is performed in batches for better memory behavior
- The implementation avoids crashing on empty clusters by keeping the previous center
- The code prints detailed cluster-level statistics, which helps when writing a report

---

## Limitations

This implementation is intentionally focused and simple.

Current limitations include:

- No plotting of cluster centers or sample images
- No dimensionality reduction for visualization
- No repeated random restarts to search for a better local optimum
- No implementation yet of K-medoids or hierarchical clustering
- No automated experimental comparison across many configurations

---

## Possible Extensions

Possible next steps for extending the repository:

- add **K-medoids** for histogram features
- add **hierarchical clustering**
- compare all clustering methods on the same balanced subsets
- export results to CSV or Markdown tables
- visualize representative samples per cluster
- visualize cluster centers for `R1`
- run multiple seeds and report mean performance

---

## Academic Context

This project was developed in the context of a university assignment on **Machine Learning** and **data clustering**, using the Fashion-MNIST dataset as the experimental benchmark.

The assignment emphasizes:

- implementation of clustering methods from scratch,
- experimentation with multiple feature representations,
- comparison of different distance measures,
- and evaluation using clustering quality metrics.

---

## Author

Replace with your personal details:

```text
Name Surname
Department of Computer Engineering and Informatics
University of Ioannina
```

---

## License

This repository is shared for educational purposes.

If you want, you can later add an explicit license such as `MIT`.
