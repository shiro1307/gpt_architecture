import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
we have a batch of blocks,
and those blocks are randomly sampled.

blocks have characters/tokens.
"""

# B -> batch size
# T -> block size
# C -> embedding dimension
# V - > vocabulary size

class MultiHeadAttention(nn.Module):
    def __init__(self, C, num_heads, block_size, dropout):
        super().__init__()

        assert C % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = C // num_heads

        # single fused projection for q, k, v across all heads at once
        self.qkv = nn.Linear(C, 3 * C, bias=False)

        self.proj = nn.Linear(C, C)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # compute q, k, v for all heads in one matmul
        q, k, v = self.qkv(x).split(C, dim=2)  # each: (B, T, C)

        # reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores: (B, num_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax + dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # combine: (B, num_heads, T, head_dim) -> (B, T, C)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        out = self.resid_dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
  def __init__(self,C, dropout):
    super().__init__()

    #expand c -> 4c
    self.fc1 = nn.Linear(C, 4*C)

    # nn.Linear is a layer of neurons.

    #apply gelu
    self.activation = nn.GELU()

    #project 4c -> c
    self.fc2 = nn.Linear(4*C,C)

    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    #pass x thru layers
    x = self.fc1(x)
    x = self.activation(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, C, num_heads, block_size, dropout):
    super().__init__()
    self.ln1 = nn.LayerNorm(C)
    self.ln2 = nn.LayerNorm(C)
    self.attention = MultiHeadAttention(C, num_heads, block_size, dropout)
    self.feed_forward = FeedForward(C, dropout)

    #store the components here


  def forward(self, x):
    #connect components here

    #residual connection, add back input
    x = x + self.attention(self.ln1(x))

    x = x + self.feed_forward(self.ln2(x))

    return x

class GPTLanguageModel(nn.Module):
  def __init__(self,vocab_size, C,num_heads,block_size, num_blocks, dropout=0.2):
    super().__init__()

    self.block_size = block_size

    #converts each token id into a vector of size C
    self.embed = nn.Embedding(vocab_size, C)

    #positional embedding
    self.pembed = nn.Embedding(block_size, C)

    self.embedding_dropout = nn.Dropout(dropout)

    #transformer blocks
    self.blocks = nn.Sequential()
    for n in range(num_blocks):

      block = Block(C, num_heads, block_size, dropout)

      self.blocks.append(block)
    
    #final layernorm
    self.lnorm = nn.LayerNorm(C)

    #language model head
    self.head = nn.Linear(C, vocab_size)

    # weight tying: share embedding and output projection weights
    # reduces parameters and improves generalization
    self.head.weight = self.embed.weight
    
  def forward(self, idx):
    B,T = idx.shape 

    #token embeddings
    token_embed = self.embed(idx)

    #positional embeddings
    pos = torch.arange(T, device=idx.device)
    pos_embed = self.pembed(pos)

    #add them together
    total_embed = token_embed + pos_embed
    total_embed = self.embedding_dropout(total_embed)

    #pass through transformer blocks
    transf_out = self.blocks(total_embed)

    #final layernorm
    normed_out = self.lnorm(transf_out)

    #language model head
    logits = self.head(normed_out)

    return logits

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature = 0.8):
    for _ in range(max_new_tokens):
        # Keep only the most recent block_size tokens
        idx_cond = idx[:, -self.block_size:]

        #Predict the next character
        logits = self(idx_cond)

        # Get logits for the last position
        logits = logits[:, -1, :]

        # Convert scores to probabilities
        probs = F.softmax(logits / temperature, dim=-1) 

        # Sample the next character
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append it to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx