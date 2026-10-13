import torch
import torch.nn.functional as F
import math

from model import GPTLanguageModel
from tokenizer import train_data, val_data, get_batch, vocab_size, decode

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training on:", device)
train_data = train_data.to(device)
val_data = val_data.to(device)

def calculate_loss(logits, targets):
    B, T, V = logits.shape

    #view reshapes a tensor
    logits = logits.view(B*T, V)
    targets = targets.view(B*T)

    return F.cross_entropy(logits, targets)

eval_iters = 20  # number of batches to average over during evaluation

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data)
    ]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            losses[k] = calculate_loss(logits, y).item()

        results[split] = losses.mean().item()

    model.train()
    return results


"""
batch_size = 64
block_size = 256
C = 384
num_heads = 6
num_blocks = 6

learning_rate = 1e-3
dropout = 0.2
iters = 5000
"""

batch_size = 64
block_size = 128
C = 256
num_heads = 4
num_blocks = 6
learning_rate = 1e-3

dropout = 0.2

model = GPTLanguageModel(
    vocab_size, C, num_heads, block_size, num_blocks, dropout=dropout
).to(device)

# torch.compile fuses ops into optimized kernels — 2-3x faster on GPU (PyTorch >= 2.0)
if hasattr(torch, "compile"):
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

iters = 5000

min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = iters

def get_lr(step):
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters

    if step >= lr_decay_iters:
        return min_lr

    decay_ratio = (step - warmup_iters) / (
        lr_decay_iters - warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)

print(f"Starting training for {iters} steps...")

for i in range(iters):

    lr = get_lr(i)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch(train_data, batch_size, block_size)

    logits = model(x)
    loss = calculate_loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if i % 500 == 0:
        print(f"Step {i}: training loss = {loss.item():.4f}, LR = {lr:.6f}")

    if i % 1000 == 0 or i == iters - 1:
        losses = estimate_loss()
        print(
            f"Step {i}: "
            f"Train loss = {losses['train']:.4f}, "
            f"Val loss = {losses['val']:.4f}"
        )

torch.save({
    "model_state_dict": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "C": C,
    "num_heads": num_heads,
    "num_blocks": num_blocks,
    "block_size": block_size,
    "dropout": dropout,
}, "gpt_checkpoint.pth")

print("model has been saved")