# DeepSeek‑Style Sparse Mixture of Experts (MoE) Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

**A clean, educational PyTorch implementation of a Transformer with Sparse Mixture of Experts (MoE), inspired by the DeepSeek family of models.**

This repository provides a from‑scratch implementation of a character‑level language model that combines multi‑head self‑attention with a sparse MoE layer featuring noisy top‑k routing. It is designed to be readable, educational, and easily extensible.

---

## 🎥 Live Demo

Run the full implementation directly in your browser – no install, no GPU needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qc4u1hOYf3comFQiRerkqRn9EQwTcaGu)

The notebook will:
1. Download the training text automatically from this repository.
2. Train the sparse MoE transformer on‑the‑fly.
3. Generate a sample of 500 characters after training.

You can also change the hyperparameters or upload your own dataset to experiment.





## 🏗️ Architecture Overview

```
Input Tokens (idx)
        │
        ▼
┌───────────────────────┐
│ Token + Position Embed │
└───────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│         Transformer Block × n_layer   │
│  ┌─────────────────────────────────┐  │
│  │  Multi‑Head Self‑Attention      │  │
│  │  + Residual + LayerNorm         │  │
│  └─────────────────────────────────┘  │
│                    │                   │
│                    ▼                   │
│  ┌─────────────────────────────────┐  │
│  │  Sparse Mixture of Experts      │  │
│  │  (Noisy Top‑k Routing)          │  │
│  │  + Residual + LayerNorm         │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Final LayerNorm       │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ LM Head (Linear)      │
└───────────────────────┘
        │
        ▼
   Output Logits
```

### Key Components

| Component | Description |
|-----------|-------------|
| **`NoisyTopkRouter`** | Routes each token to its top‑k experts. Adds learnable Gaussian noise to the router logits to encourage load balancing. |
| **`SparseMoE`** | Applies the selected expert networks to each token and aggregates the weighted outputs. |
| **`Expert`** | A simple 2‑layer MLP with ReLU activation and dropout (4× expansion factor). |
| **`Head`** | A single head of scaled dot‑product self‑attention with causal masking. |
| **`MultiHeadAttention`** | Parallel self‑attention heads with a final projection layer. |
| **`Block`** | A transformer block combining attention and MoE with residual connections and layer norm. |
| **`SparseMoELanguageModel`** | The full language model: embeddings, stacked blocks, final layer norm, and language modelling head. |

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 16 | Number of sequences processed in parallel |
| `block_size` | 32 | Maximum context length (sequence length) |
| `max_iters` | 50,000 | Total training iterations |
| `eval_interval` | 100 | How often to evaluate on validation set |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `n_embed` | 128 | Embedding dimension |
| `n_head` | 8 | Number of self‑attention heads |
| `n_layer` | 8 | Number of transformer blocks |
| `dropout` | 0.1 | Dropout probability |
| `num_experts` | 8 | Total number of experts in the MoE layer |
| `top_k` | 2 | Number of experts activated per token |
| `eval_iters` | 400 | Number of batches used for evaluation loss |

---

## 🚀 Quick Start

just open colab and enjoy the project
