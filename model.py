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

class Head(nn.Module):
  def __init__(self, C, head_dim):
    super().__init__()

    self.key = nn.Linear(C,head_dim,bias = False)
    self.query = nn.Linear(C,head_dim,bias = False)
    self.value = nn.Linear(C,head_dim,bias = False)

  def forward(self,x):
    B,T,C = x.shape

    #1. project x into q, k, v
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    d_k = q.size(-1) 

    #2. calc attention scores
    scores = torch.matmul(q, k.transpose(-2, -1))

    #3. scale scores
    scaled_scores = scores / math.sqrt(d_k)

    #4. apply casual mask
    mask = torch.tril(torch.ones(T, T, device=x.device))
    scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
    
    #5. apply softmax
    attention_weights = F.softmax(scaled_scores, dim=-1) 

    #6. combine attention weights with values
    output = torch.matmul(attention_weights, v) 

    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, C, num_heads, block_size):
        super().__init__()

        head_dim = C // num_heads

        #holds neural modules in a trackable list
        self.heads = nn.ModuleList()
        
        assert C % num_heads == 0

        #create multiple Head instances
        for n in range(num_heads):
          head = Head(C,head_dim)
          self.heads.append(head)

        #create final linear projection
        self.proj = nn.Linear(C,C)

    def forward(self, x):
        #run every head on x
        head_outputs = [head(x) for head in self.heads]

        #concatenate outputs along last dimension
        out = torch.cat(head_outputs, dim=-1)

        #apply final projection
        out = self.proj(out)

        return out

class FeedForward(nn.Module):
  def __init__(self,C):
    super().__init__()

    #expand c -> 4c
    self.fc1 = nn.Linear(C, 4*C)

    # nn.Linear is a layer of neurons.

    #apply gelu
    self.activation = nn.GELU()

    #project 4c -> c
    self.fc2 = nn.Linear(4*C,C)

  def forward(self,x):
    #pass x thru layers
    x = self.fc1(x)
    x = self.activation(x)
    out = self.fc2(x)
    return out

class Block(nn.Module):
  def __init__(self, C, num_heads, block_size):
    super().__init__()
    self.ln1 = nn.LayerNorm(C)
    self.ln2 = nn.LayerNorm(C)
    self.attention = MultiHeadAttention(C, num_heads, block_size)
    self.feed_forward = FeedForward(C)

    #store the components here


  def forward(self, x):
    #connect components here

    #residual connection, add back input
    x = x + self.attention(self.ln1(x))

    x = x + self.feed_forward(self.ln2(x))

    return x

class GPTLanguageModel(nn.Module):
  def __init__(self,vocab_size, C,num_heads,block_size, num_blocks):
    super().__init__()

    #converts each token id into a vector of size C
    self.embed = nn.Embedding(vocab_size, C)

    #positional embedding
    self.pembed = nn.Embedding(block_size, C)

    #transformer blocks
    self.blocks = nn.Sequential()
    for n in range(num_blocks):
      block = Block(C, num_heads, block_size)
      self.blocks.append(block)
    
    #final layernorm
    self.lnorm = nn.LayerNorm(C)

    #language model head
    self.head = nn.Linear(C, vocab_size)
    
  def forward(self, idx):
    B,T = idx.shape 

    #token embeddings
    token_embed = self.embed(idx)

    #positional embeddings
    pos = torch.arange(T, device=idx.device)
    pos_embed = self.pembed(pos)

    #add them together
    total_embed = token_embed + pos_embed

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
        idx_cond = idx[:, -self.pembed.num_embeddings:]

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