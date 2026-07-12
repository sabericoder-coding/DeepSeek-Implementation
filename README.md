# DeepSeek From Scratch 




**a minimal educational implementation of deepseek that only covers the absolute basics. Here is the actual feature list of code**

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

## 🎥 Live Demo

Run the full implementation directly in your browser – no install, no GPU needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qc4u1hOYf3comFQiRerkqRn9EQwTcaGu)

The notebook will:
1. Download the training text automatically from this repository.
2. Train the sparse MoE transformer on‑the‑fly.
3. Generate a sample of 500 characters after training.

You can also change the hyperparameters or upload your own dataset to experiment.

---

## 🏗️ Architecture Overview

```text
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
