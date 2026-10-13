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
checkpoint = torch.load(
"gpt_checkpoint.pth",
map_location="cpu",
weights_only=True
)

#recreate model architecture
model = GPTLanguageModel(
checkpoint["vocab_size"],
checkpoint["C"],
checkpoint["num_heads"],
checkpoint["block_size"],
checkpoint["num_blocks"]
)

#load trained weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

#Choose starting prompt
start_text = "To be, or not to be, that is the question."
context = torch.tensor([encode(start_text)], dtype=torch.long)

#generate text
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=500)

print(decode(generated[0].tolist()))