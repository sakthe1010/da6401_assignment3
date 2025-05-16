# dataset.py

import os
import torch
from torch.utils.data import Dataset

class TransliterationDataset(Dataset):
    def __init__(self, file_path):
        self.pairs = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2: continue
                self.pairs.append((parts[0], parts[1]))

        # Build vocab
        self.src_char2idx = {"<pad>":0, "<sos>":1, "<eos>":2}
        self.trg_char2idx = {"<pad>":0, "<sos>":1, "<eos>":2}

        for src, trg in self.pairs:
            for ch in src:
                if ch not in self.src_char2idx:
                    self.src_char2idx[ch] = len(self.src_char2idx)
            for ch in trg:
                if ch not in self.trg_char2idx:
                    self.trg_char2idx[ch] = len(self.trg_char2idx)

        self.src_idx2char = {i: c for c, i in self.src_char2idx.items()}
        self.trg_idx2char = {i: c for c, i in self.trg_char2idx.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]

        src_seq = [self.src_char2idx["<sos>"]] + \
                  [self.src_char2idx[c] for c in src] + \
                  [self.src_char2idx["<eos>"]]

        trg_seq = [self.trg_char2idx["<sos>"]] + \
                  [self.trg_char2idx[c] for c in trg] + \
                  [self.trg_char2idx["<eos>"]]

        return torch.tensor(src_seq), torch.tensor(trg_seq)

    def get_vocab_sizes(self):
        return len(self.src_char2idx), len(self.trg_char2idx)

    def get_pad_idx(self):
        return self.src_char2idx["<pad>"], self.trg_char2idx["<pad>"]

def collate_fn(batch):
    src_seqs, trg_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    trg_lens = [len(t) for t in trg_seqs]

    max_src = max(src_lens)
    max_trg = max(trg_lens)

    batch_size = len(batch)
    src_pad = torch.zeros((max_src, batch_size), dtype=torch.long)
    trg_pad = torch.zeros((max_trg, batch_size), dtype=torch.long)

    for i in range(batch_size):
        src_pad[:src_lens[i], i] = src_seqs[i]
        trg_pad[:trg_lens[i], i] = trg_seqs[i]

    return src_pad, trg_pad
