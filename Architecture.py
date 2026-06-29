import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
# ======================
# Hyperparameters
# ======================
batch_size = 16
block_size = 32
max_iters = 50000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 400
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8
top_k = 2
head_size = n_embed // n_head  # Calculated based on n_embed and n_head

# ======================
# Set device to TPU
'''
import torch_xla
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f"Using device: {device}")
'''
# ======================



# ======================
# Data Preparation
# ======================
torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]



# =====================
# Move data to tpu

'''
# Move data to TPU after splitting
train_data = data[:n].to(device)
val_data = data[n:].to(device)
'''
# =====================





# ======================
# Model Components
# ======================
class Expert(nn.Module):
    """Simple MLP Expert"""
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
def forward(self, x):
        return self.net(x)


#Changing the above to accomodate noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        
        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices



class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1) 
            flat_mask = expert_mask.view(-1) #Ensures the mask is a flat 1D tensor.

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)


                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


Step 9: Code the entire transformer block: Part 1 (Multi-head attention)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  
        """ 
        The line self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) creates a fixed, lower-triangular mask.
        This mask is then used in the forward method to prevent attention heads from attending to future tokens in the sequence.
        This is essential for autoregressive models (like GPT) where the model should only use past information to predict the next token.
        """

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        # x is the input tensor, typically of shape (Batch Size, Sequence Length, Embedding Dimension)
        # Example: x.shape = (B, T, C) where B=batch_size, T=sequence_length, C=embedding_dimension (n_embed)
        """
        
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out




#Multi-Headed Self Attention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



Step 10: Code the entire transformer block: Part 2 (Assemble all layers)
#First create a self attention + mixture of experts block, that may be repeated several number of times
#Copy pasting key architecture variables for clarity

class Block(nn.Module):
    """ Mixture of Experts Transformer block: communication followed by computation (multi-head self attention)
    """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x



Step 11: Define entire language model architecture

#Finally putting it all together to crease a sparse mixture of experts language model
class SparseMoELanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts,top_k=top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # B = batch_size, T = sequence_length of the input `idx`
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)  #This function creates a 1-dimensional tensor (a list) of sequential integers.
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape #B -->batch_size # T --> sequence_length # C --> vocab_size 
            logits = logits.view(B*T, C) # reshaping The B and T dimensions are multiplied together (B*T), creating a single, long dimension.The vocab_size dimension (C) remains as is.
            targets = targets.view(B*T)  # reshaping, This reshape flattens it into a single dimension.This creates a 1D tensor containing all the target token IDs, one for each of the B*T positions.
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]   #block_size (or often context_window) refers to the maximum number of tokens the model can consider at once.
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# ======================
# Training Utilities
# ======================
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    """
    [data[i:i+block_size] for i in ix]: This is a list comprehension. It iterates through each starting index i in the ix tensor. For each i,
    it slices the data tensor to grab a contiguous chunk of block_size tokens: data[i:i+block_size].
    If data looks like [10, 20, 30, 40, 50, 60, 70, ...] and block_size is 3, and ix contains [0, 3]:
    For i = 0, it gets data[0:3] which is [10, 20, 30].
    For i = 3, it gets data[3:6] which is [40, 50, 60].
    So, the list comprehension produces a list of tensors: [ tensor([10, 20, 30]), tensor([40, 50, 60]) ].
       
    """
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

Step 13: Define LLM Loss

    @torch.no_grad()
    def estimate_loss():
        out = {}  # This dictionary  stores the average losses for each data split (‘train’ and ‘val’).
        model.eval() # Sets the model to evaluation mode.
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)  #Tensor Initialization: Inside the loop for each split, a tensor named losses is created. It’s filled with zeros and has a size equal to eval_iters. This tensor will store the loss calculated over eval_iters batches.
            for k in range(eval_iters): # Batch Iteration: This inner loop runs eval_iters times. This means the loss will be averaged over eval_iters different batches of data. This is a common practice to get a more stable estimate of the loss than just using a single batch.
                X, Y = get_batch(split)
                logits, loss = model(X, Y) # Model Forward Pass: The input batch X and target batch Y are passed to the model.The model performs its forward pass.It returns:logits: The raw output scores from the model’s final layer (as discussed in previous turns).loss: The calculated loss value for this specific batch.
                losses[k] = loss.item() # Storing Loss: loss.item() extracts the scalar value from the loss tensor.This scalar loss value is then stored in the losses tensor at the k-th position.
            out[split] = losses.mean() # Averaging Loss: After iterating through all eval_iters batches for a given split: losses.mean() calculates the average of all the stored batch losses.This average loss is stored in the out dictionary with the split name (‘train’ or ‘val’) as the key.
        model.train() # Back to Training Mode: After calculating the losses for both splits, this line sets the model back to training mode. This is crucial so that layers like Dropout and Batch Normalization behave correctly for subsequent training steps.
        return out


Step 15: Initialize the entire model

def kaiming_init_weights(m):
        if isinstance(m, (nn.Linear)): # “Is this part m a ‘Linear’ type of component?”.  nn.Linear means fully connected layer 
            init.kaiming_normal_(m.weight) # init.kaiming_normal_ is the specific way to set m.weight

# ======================
# Model Initialization
# ======================
model = SparseMoELanguageModel(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    num_experts=num_experts,
    top_k=top_k,
    dropout=dropout
)
model.apply(kaiming_init_weights)
model = model.to(device)
# Print model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params/1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# ======================
# Training Loop
# ======================
for iter in range(max_iters):
    # Evaluation step
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
# Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# ======================
# Text Generation
# ======================
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
