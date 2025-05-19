import torch
from torch.utils.data import Dataset
import unicodedata
import os

def normalize(text):
    return unicodedata.normalize("NFC", text.strip())

def read_data(file_path):
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    tgt = normalize(parts[0])
                    src = normalize(parts[1])
                    pairs.append((src, tgt))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")

    print(f"[INFO] Loaded {len(pairs)} pairs from {file_path}")
    return pairs

def build_vocab(pairs, sos='<sos>', eos='<eos>', pad='<pad>'):
    vocab = set(char for seq, _ in pairs for char in seq)
    vocab |= set(char for _, seq in pairs for char in seq)
    tokens = [pad, sos, eos] + sorted(vocab)
    idx2char = tokens
    char2idx = {ch: idx for idx, ch in enumerate(tokens)}
    return char2idx, idx2char

def encode(seq, char2idx, max_len):
    tokens = ['<sos>'] + list(seq) + ['<eos>']
    idxs = [char2idx.get(c, 0) for c in tokens]
    idxs += [char2idx['<pad>']] * (max_len - len(idxs))
    return idxs[:max_len]

class TransliterationDataset(Dataset):
    def __init__(self, data_pairs, inp_char2idx, out_char2idx, max_len_input, max_len_output):
        self.data_pairs = data_pairs
        self.inp_char2idx = inp_char2idx
        self.out_char2idx = out_char2idx
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src, tgt = self.data_pairs[idx]
        src_tensor = torch.tensor(encode(src, self.inp_char2idx, self.max_len_input))
        tgt_tensor = torch.tensor(encode(tgt, self.out_char2idx, self.max_len_output))
        return src_tensor, tgt_tensor
