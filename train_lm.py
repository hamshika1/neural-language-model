import os
import math
import random
import time
import json
from collections import Counter
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------
SEED = 12345
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

DATA_PATH = "data/dataset.txt"
OUT_DIR = "outputs"

# ---------------------------------------
# TOKENIZATION + VOCAB
# ---------------------------------------
class Vocab:
    def __init__(self, min_freq=1):
        self.specials = ["<pad>", "<unk>", "<eos>"]
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = []

    def build(self, token_list):
        counter = Counter(token_list)
        words = [w for w, c in counter.items() if c >= self.min_freq]
        self.itos = self.specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        self.pad = self.stoi["<pad>"]
        self.unk = self.stoi["<unk>"]
        self.eos = self.stoi["<eos>"]

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]

# ---------------------------------------
# DATASET
# ---------------------------------------
class LMDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.ids[idx: idx + self.seq_len]
        y = self.ids[idx + 1: idx + 1 + self.seq_len]
        return torch.tensor(x), torch.tensor(y)

# ---------------------------------------
# MODELS
# ---------------------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=128, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        o, _ = self.lstm(e)
        o = self.drop(o)
        return self.fc(o)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, emb=256, heads=8, layers=4, ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.pos = nn.Embedding(max_len, emb)

        enc = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(emb, vocab_size)

    def forward(self, x):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        e = self.emb(x) + self.pos(pos)
        e = e.transpose(0, 1)

        mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
        o = self.enc(e, mask=mask)
        o = o.transpose(0, 1)
        return self.fc(o)

# ---------------------------------------
# TRAINING + EVAL
# ---------------------------------------
def evaluate(model, loader, crit):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

def train_epoch(model, loader, optim, crit):
    model.train()
    total_loss = 0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        logits = model(x)
        loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

# ---------------------------------------
# RUN EXPERIMENT (one config)
# ---------------------------------------
def run_experiment(name, config):

    print(f"\n=== Running: {name} ===")

    # Load dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip().split() + ["<eos>"] for l in f.readlines()]

    all_tokens = [t for line in lines for t in line]

    # Build vocab
    vocab = Vocab(min_freq=config["min_freq"])
    vocab.build(all_tokens)

    # Convert to ids
    ids = vocab.encode(all_tokens)

    # Split train/val/test
    n = len(ids)
    train_ids = ids[: int(0.8*n)]
    val_ids = ids[int(0.8*n): int(0.9*n)]
    test_ids = ids[int(0.9*n):]

    # Datasets
    train_ds = LMDataset(train_ids, config["seq_len"])
    val_ds   = LMDataset(val_ids,   config["seq_len"])
    test_ds  = LMDataset(test_ids,  config["seq_len"])

    train_ld = DataLoader(train_ds, batch_size=config["batch"], shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=config["batch"])
    test_ld  = DataLoader(test_ds,  batch_size=config["batch"])

    # Model
    if config["model"] == "lstm":
        model = LSTMLM(len(vocab.itos),
                       emb=config["emb"],
                       hid=config["hid"],
                       layers=config["layers"])
    else:
        model = TransformerLM(len(vocab.itos),
                              emb=config["emb"],
                              heads=config["heads"],
                              layers=config["layers"],
                              ff=config["ff"])

    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training
    history = {"train": [], "val": []}

    os.makedirs(f"{OUT_DIR}/{name}", exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_ppl = train_epoch(model, train_ld, optim, crit)
        va_loss, va_ppl = evaluate(model, val_ld, crit)

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        print(f"Epoch {epoch}/{config['epochs']} | "
              f"Train Loss={tr_loss:.3f} PPL={tr_ppl:.2f} | "
              f"Val Loss={va_loss:.3f} PPL={va_ppl:.2f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), f"{OUT_DIR}/{name}/best_model.pt")

    # Test
    test_loss, test_ppl = evaluate(model, test_ld, crit)
    print(f"Final Test Loss={test_loss:.3f}, Test PPL={test_ppl:.2f}")

    # Save results
    with open(f"{OUT_DIR}/{name}/results.txt", "w") as f:
        f.write(f"test_loss: {test_loss}\n")
        f.write(f"test_ppl: {test_ppl}\n")

    # Plot
    plt.figure()
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(name)
    plt.legend()
    plt.savefig(f"{OUT_DIR}/{name}/loss.png")
    plt.close()

# ---------------------------------------
# EXPERIMENT CONFIGURATIONS
# ---------------------------------------
EXPS = {
    "underfit": {
        "model": "lstm",
        "emb": 64,
        "hid": 64,
        "layers": 1,
        "lr": 1e-3,
        "batch": 64,
        "seq_len": 20,
        "epochs": 3,
        "min_freq": 2,
    },
    "overfit": {
        "model": "lstm",
        "emb": 300,
        "hid": 600,
        "layers": 3,
        "lr": 1e-4,
        "batch": 16,
        "seq_len": 50,
        "epochs": 8,
        "min_freq": 1,
    },
    "best_fit": {
        "model": "transformer",
        "emb": 256,
        "heads": 8,
        "layers": 4,
        "ff": 512,
        "lr": 5e-4,
        "batch": 64,
        "seq_len": 64,
        "epochs": 5,
        "min_freq": 2,
    }
}

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, cfg in EXPS.items():
        run_experiment(name, cfg)

if __name__ == "__main__":
    main()
