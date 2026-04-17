# 🚲 Predicting Bike Sharing Patterns — Neural Network from Scratch

![Python](https://img.shields.io/badge/Python-3.6+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.18-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.0-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🚀 Overview

A from-scratch implementation of a 2-layer neural network (no TensorFlow, no PyTorch — just NumPy) that predicts hourly bike rental demand using the Capital Bikeshare dataset (Washington D.C., 2011–2012). The network learns to capture complex patterns in weather, time-of-day, seasonality, and day-of-week features to forecast ridership counts.

This project demonstrates a ground-up understanding of how neural networks work — forward propagation, backpropagation, gradient descent, and hyperparameter tuning — without relying on any deep learning framework.

## ✨ Key Features

- **Neural Network from Scratch** — Complete `NeuralNetwork` class with forward pass, backpropagation, and weight updates implemented using only NumPy matrix operations
- **Sigmoid Activation + Linear Output** — Hidden layer uses sigmoid activation; output layer uses identity function (f(x) = x) for regression
- **Stochastic Gradient Descent** — Training uses random mini-batches of 128 samples per iteration for efficient convergence
- **Feature Engineering Pipeline** — One-hot encoding of 5 categorical variables (season, weather, month, hour, weekday), z-score normalization of 6 continuous variables, and train/validation/test split on time-series data
- **Hyperparameter Tuning** — Tuned to 1600 iterations, learning rate of 1.57, and 9 hidden nodes — balancing underfitting vs. overfitting
- **Unit Tests** — Built-in tests verify sigmoid activation correctness, weight update accuracy (to 8 decimal places), and forward pass output

## 🧠 Technical Highlights

- **Manual Backpropagation** — Computes output error, propagates it backward through the hidden layer using the chain rule, and accumulates weight gradients across the batch before applying a single averaged update. The output error term uses a derivative of 1 (identity activation), while the hidden error term applies the sigmoid derivative: `hidden_outputs * (1 - hidden_outputs)`
- **Weight Initialization** — Uses scaled random normal initialization (`np.random.normal(0, n^-0.5)`) where n is the number of input nodes, preventing vanishing/exploding gradients at initialization
- **Data Preprocessing** — Categorical features are one-hot encoded via `pd.get_dummies()`, producing 56 input features from the original 17 columns. Continuous features are z-score standardized (zero mean, unit variance) with scaling factors saved for inverse-transforming predictions back to original scale
- **Time-Series Aware Splitting** — Last 21 days reserved for testing, preceding 60 days for validation, remainder for training — no data leakage from future to past

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.6+ |
| Core Math | NumPy |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |
| Testing | unittest |

## 🏗 Architecture / How It Works

```
Input Layer (56 nodes)          Hidden Layer (9 nodes)         Output Layer (1 node)
┌─────────────────────┐        ┌──────────────────┐          ┌─────────────────┐
│                     │        │                  │          │                 │
│  56 features        │──W1──▶ │  Sigmoid(W1·X)   │───W2───▶ │  f(x) = x      │
│  (one-hot encoded   │        │                  │          │  (bike count)   │
│   + normalized)     │        │  9 hidden nodes   │          │                 │
│                     │        │                  │          │                 │
└─────────────────────┘        └──────────────────┘          └─────────────────┘

W1: 56×9 matrix (input→hidden)     W2: 9×1 matrix (hidden→output)
Activation: sigmoid                 Activation: identity (regression)
```

**Training Loop (per iteration):**
1. Sample random mini-batch of 128 records from training data
2. **Forward pass:** compute hidden activations (sigmoid) and output (linear)
3. **Backward pass:** compute output error → backpropagate to hidden layer → accumulate weight gradients
4. **Update:** apply averaged gradients scaled by learning rate

**Data Pipeline:**
```
Raw CSV (17,379 hourly records, 17 columns)
    │
    ▼
One-hot encode: season, weathersit, month, hour, weekday → 56 features
    │
    ▼
Z-score normalize: temp, humidity, windspeed, casual, registered, cnt
    │
    ▼
Split: Train (historical) → Validation (60 days) → Test (21 days)
    │
    ▼
Train network → Predict → Inverse-transform → Compare with actual
```

## ⚡ Getting Started

### Prerequisites

- Python 3.6+
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/jashjain21/Predicting-Bike-Sharing-Patterns.git
cd Predicting-Bike-Sharing-Patterns

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Your_first_neural_network.ipynb
```

### Running as a Script

```bash
python Your_first_neural_network.py
```

## 📌 Example Usage

The notebook walks through the full pipeline:

1. **Load data** — 17,379 hourly records from `Bike-Sharing-Dataset/hour.csv`
2. **Explore** — Visualize ridership patterns over the first 10 days (weekday commute spikes, weekend dips)
3. **Preprocess** — One-hot encode categoricals, normalize continuous features, split into train/val/test
4. **Train** — Run 1600 iterations of SGD with mini-batches of 128
5. **Evaluate** — Plot training vs. validation loss curves to check for overfitting
6. **Predict** — Compare predicted vs. actual ridership for the final 21 days

**Tuned Hyperparameters:**
| Parameter | Value |
|---|---|
| Iterations | 1600 |
| Learning Rate | 1.57 |
| Hidden Nodes | 9 |
| Output Nodes | 1 |
| Batch Size | 128 |

## 🔍 What This Project Demonstrates

- **Neural Network Fundamentals** — Forward propagation, backpropagation, and gradient descent implemented from first principles with NumPy, not abstracted away by a framework
- **Linear Algebra in Practice** — Matrix multiplication for layer computations, element-wise operations for activation functions, and outer products for gradient accumulation
- **Feature Engineering** — One-hot encoding, z-score normalization, and inverse-transformation for interpretable predictions
- **Hyperparameter Tuning** — Systematic selection of learning rate, hidden nodes, and iterations by monitoring train/validation loss divergence
- **Time-Series Data Handling** — Chronological train/validation/test splits that prevent data leakage

## 🚧 Limitations / Future Improvements

- **Single Hidden Layer** — The network has only one hidden layer with 9 nodes; adding depth could capture more complex temporal patterns
- **No Regularization** — No dropout or L2 regularization is applied; the model relies solely on early stopping via iteration count to prevent overfitting
- **Holiday Prediction Gap** — The model struggles during the last week of December (Christmas/New Year) where ridership patterns deviate significantly from normal — likely due to insufficient holiday training examples
- **No Learning Rate Scheduling** — A fixed learning rate of 1.57 is used throughout; a decaying schedule could improve convergence in later iterations
- **No Cross-Validation** — A single train/validation split is used; k-fold cross-validation would give more robust hyperparameter estimates
- **Batch Normalization** — Not implemented; adding it between layers could stabilize training and allow higher learning rates

## 📊 Dataset

The [Capital Bikeshare dataset](http://capitalbikeshare.com/system-data) contains 17,379 hourly records from 2011–2012 with 17 features including weather conditions, time, and ridership counts. Dataset provided by Fanaee-T & Gama (2013), University of Porto.
