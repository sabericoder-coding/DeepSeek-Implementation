# DeepSeek‑Style Sparse Mixture of Experts (MoE) Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

**A clean, educational PyTorch implementation of a Transformer with Sparse Mixture of Experts (MoE), inspired by the DeepSeek family of models.**

This repository provides a from‑scratch implementation of a character‑level language model that combines multi‑head self‑attention with a sparse MoE layer featuring noisy top‑k routing. It is designed to be readable, educational, and easily extensible.

---

## 🎥 Live Demo

Run the full implementation directly in your browser – no install, no GPU needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID)

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

### Prerequisites

```bash
pip install torch numpy
```

### Prepare Your Data

Place your training text in a file named `input.txt` in the project root. The script will read this file and build a character‑level vocabulary.

### Train the Model

```bash
python Architecture.py
```

The script will:
1. Load and preprocess the text data
2. Initialize the MoE language model
3. Train for 50,000 iterations
4. Print training and validation loss every 100 steps
5. Generate 500 new characters after training

### Example Output

```
14.23M parameters
step 0: train loss 4.2345, val loss 4.2987
step 100: train loss 2.1234, val loss 2.1987
...
[Generated text output]
```

---

## 🧠 Key Implementation Details

### Noisy Top‑k Routing

The router adds Gaussian noise to its logits before selecting the top‑k experts. This encourages a more balanced distribution of tokens across experts and prevents routing collapse.

```python
noise = torch.randn_like(logits) * F.softplus(noise_logits)
noisy_logits = logits + noise
top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
```

### Sparse Expert Computation

Only the selected experts are actually computed, saving significant FLOPs compared to a dense model. Each expert processes its assigned tokens independently, and the results are aggregated using the router's gating scores.

```python
for i, expert in enumerate(self.experts):
    expert_mask = (indices == i).any(dim=-1)
    if flat_mask.any():
        expert_input = flat_x[flat_mask]
        expert_output = expert(expert_input)
        gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
        weighted_output = expert_output * gating_scores
        final_output[expert_mask] += weighted_output.squeeze(1)
```

### Causal Self‑Attention

Each attention head uses a lower‑triangular mask to prevent attending to future tokens, ensuring the model is autoregressive.

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

---

## 📁 Project Structure

```
DeepSeek-Implementation/
├── Architecture.py      # Full model implementation and training script
├── input.txt            # Training data (Shakespeare corpus)
└── README.md            # This file
```

---

## 📚 References

- **DeepSeek-V2**: *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* — Introduces the efficient MoE architecture that inspired this implementation.
- **Switch Transformer**: *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* — Foundation for sparse routing in Transformers.
- **Andrej Karpathy's nanoGPT**: A minimalist GPT implementation that served as a reference for the transformer backbone.

---

## 🔮 Future Work

- [ ] Add expert capacity and load‑balancing auxiliary loss
- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Support multi‑GPU training with expert parallelism
- [ ] Add learning rate warmup and cosine decay scheduling
- [ ] Integrate with Hugging Face `transformers` for easy loading

---

## 📝 License

MIT — Free for academic and portfolio use.

---

**Built as part of my research into efficient sparse architectures and Mixture of Experts systems.**
