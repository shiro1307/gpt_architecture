import torch
import torch.nn.functional as F

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

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data)
    ]:
        losses = torch.zeros(100)

        for k in range(100):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            losses[k] = calculate_loss(logits, y).item()

        results[split] = losses.mean().item()

    model.train()
    return results


batch_size = 64
block_size = 128
C = 128
num_heads = 4
num_blocks = 4

model = GPTLanguageModel(
    vocab_size, C, num_heads, block_size, num_blocks
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

iters = 10000

for i in range(iters):
    x, y = get_batch(train_data, batch_size, block_size)

    logits = model(x)
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f"Step {i}: training loss = {loss.item():.4f}")

    if i % 1000 == 0:
        losses = estimate_loss()
        print(
            f"Step {i}: "
            f"Train loss = {losses['train']:.4f}, "
            f"Validation loss = {losses['val']:.4f}"
        )

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab_size": vocab_size,
    "C": C,
    "num_heads": num_heads,
    "num_blocks": num_blocks,
    "block_size": block_size,
}, "gpt_checkpoint.pth")

print("model has been saved")