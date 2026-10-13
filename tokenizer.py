import torch

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda s : [stoi[c] for c in s]
decode = lambda nums : ''.join(itos[i] for i in nums)

data = torch.tensor(encode(text), dtype=torch.long)

split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]

#batch size = how many blocks/sequences the model processes at once
#block size = how many chars the model processes in one sequence
# x and y's dim = [batch size , block size]

#batch has blocks.
#blocks have chars/tokens.

# X and Y are just a batch

def get_batch(data, batch_size, block_size):

    """
    we have a batch of blocks,
    and those blocks are randomly sampled.

    blocks have characters/tokens.
    """

    #data -> tensor with encoded text

    #contains batch_size random starting positions
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
        device=data.device
    )

    #stacking blocks together in one tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    #one character shift in target
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y

import pickle

if __name__ == "__main__":
    print(f"Dataset size: {len(text)} chars")
    print(f"Vocab size: {vocab_size}")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos}, f)

    print("tokenizer.pkl saved")