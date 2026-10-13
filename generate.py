import torch
import torch.nn.functional as F
import pickle

from model import GPTLanguageModel

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stoi = tokenizer["stoi"]
itos = tokenizer["itos"]

def encode(s):
    return [stoi[c] for c in s]

def decode(nums):
    return ''.join(itos[i] for i in nums)

#load checkpoint
try:
    # weights_only=True is safer and supported on PyTorch >= 1.13
    checkpoint = torch.load(
        "gpt_checkpoint.pth",
        map_location="cpu",
        weights_only=True
    )
except TypeError:
    # fallback for older PyTorch versions that don't support weights_only
    checkpoint = torch.load(
        "gpt_checkpoint.pth",
        map_location="cpu"
    )

#recreate model architecture
model = GPTLanguageModel(
checkpoint["vocab_size"],
checkpoint["C"],
checkpoint["num_heads"],
checkpoint["block_size"],
checkpoint["num_blocks"],
dropout=checkpoint.get("dropout", 0.2)
)

#load trained weights
state_dict = checkpoint["model_state_dict"]

# strip _orig_mod. prefix if checkpoint was saved from a torch.compile'd model
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

#Choose starting prompt
start_text = "To be, or not to be, that is the question."
context = torch.tensor([encode(start_text)], dtype=torch.long)

#generate text
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=1000)

print(decode(generated[0].tolist()))