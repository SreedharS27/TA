## Self-Pruning Neural Network (PyTorch)

## Overview

This project implements a **self-pruning neural network** that learns to remove its own unnecessary weights **during training**, instead of relying on post-processing pruning techniques.

The model is trained on the **CIFAR-10 dataset** and uses a novel mechanism where each weight is associated with a **learnable gate parameter**. These gates determine whether a connection should remain active or be pruned.

---

##  Key Features

*  Custom `PrunableLinear` layer (built from scratch)
*  Learnable gating mechanism using sigmoid
*  Automatic pruning during training
*  L1-based sparsity regularization
*  Trade-off analysis between accuracy and sparsity
*  Visualization of gate distribution

---

##  Architecture

Each weight `w` is paired with a learnable gate `g`:

[
\text{Effective Weight} = w \cdot \sigma(g)
]

Where:

* `σ(g)` is the sigmoid function (values between 0 and 1)
* If gate → 0 → weight is effectively removed

---

##  Loss Function

[
\text{Total Loss} = \text{CrossEntropyLoss} + \lambda \cdot \sum \text{gates}
]

* **CrossEntropyLoss** → classification objective
* **L1 penalty on gates** → enforces sparsity
* **λ (lambda)** → controls pruning strength

---

##  Project Structure

```
├── main.py              # Complete training & evaluation script
├── data/                # CIFAR-10 dataset (from torchvision)
├── README.md            # Project documentation
```

---

##  Installation

```bash
pip install torch torchvision matplotlib
```

---

##  Usage

The script will:

1. Train models for different λ values
2. Evaluate test accuracy
3. Compute sparsity level
4. Plot gate distribution

---


##  Observations

*  Low λ → High accuracy, low pruning
*  Medium λ → Balanced performance
*  High λ → High sparsity, lower accuracy

This demonstrates the **sparsity–accuracy trade-off**.

---

## Gate Distribution

After training, the gate values typically show:

* A spike near **0** → pruned weights
* A cluster away from 0 → important connections

This confirms successful **self-pruning behavior**.

---

## Key Concepts

* **Dynamic pruning** (during training)
* **L1 regularization for sparsity**
* **Differentiable gating mechanism**
* **Model compression**

---

