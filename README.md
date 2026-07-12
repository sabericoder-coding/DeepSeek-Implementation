# DeepSeek‑Style Sparse Mixture of Experts (MoE) Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

**
a minimal educational implementation of deepseek that only covers the absolute basics. Here is the actual feature list of code

---

**Features Implemented in This Repository**

• **Sparse Mixture-of-Experts (MoE)** – with a noisy top‑k router that selects 2 experts per token from a pool of 8.

• **Noisy Top‑k Gating** – adds learnable Gaussian noise to router logits to encourage load balancing.

• **Multi‑Head Self‑Attention (MHSA)** – standard scaled dot‑product attention with 8 heads and causal masking.

• **Pre‑Layer Normalization Transformer Blocks** – applies LayerNorm before attention and MoE, with residual connections.

• **Learned Positional Embeddings** – uses a trainable embedding table for absolute position encoding (not RoPE or ALiBi).

• **Character‑Level Tokenization** – custom encode/decode functions mapping characters to integers (not subword/Byte‑Pair Encoding).

• **Kaiming (He) Initialization** – applied to all linear layers for better training stability.

• **Train / Validation Split** – 90/10 split of the input text data with periodic evaluation loss.

• **Autoregressive Text Generation** – greedy multinomial sampling to generate new tokens sequentially.

• **Dropout Regularization** – applied in attention, expert MLPs, and final projections.

• **CUDA / CPU Support** – automatically selects GPU if available (with commented TPU support using `torch_xla`).

• **AdamW Optimizer** – used for training with a fixed learning rate.

---
**

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
