import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd


# ---------- module-level placeholders so they can be imported ----------
src_char2idx = {}
tgt_char2idx = {}
src_idx2char = {}
tgt_idx2char = {}

def get_vocabs():
    """
    Returns the four lookup tables once the first TransliterationDataset
    has been instantiated.
    """
    return src_char2idx, tgt_char2idx, src_idx2char, tgt_idx2char
# ------------------------------------------------------------------------


class TransliterationDataset(Dataset):
    def __init__(self, path, src_vocab=None, tgt_vocab=None):
        self.data = pd.read_csv(path, sep='\t', header=None,
                                names=['target', 'input', 'freq'])
        self.inputs  = self.data['input'].astype(str).tolist()
        self.targets = self.data['target'].astype(str).tolist()

        self.src_vocab = src_vocab or self.build_vocab(self.inputs)
        self.tgt_vocab = tgt_vocab or self.build_vocab(self.targets)

        self.src_idx2char = {i: ch for ch, i in self.src_vocab.items()}
        self.tgt_idx2char = {i: ch for ch, i in self.tgt_vocab.items()}

        # ---------- ➋ expose the dictionaries at module scope ----------
        global src_char2idx, tgt_char2idx, src_idx2char, tgt_idx2char
        src_char2idx = self.src_vocab
        tgt_char2idx = self.tgt_vocab
        src_idx2char = self.src_idx2char
        tgt_idx2char = self.tgt_idx2char
        # ----------------------------------------------------------------

    def build_vocab(self, sequences):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        idx = 3
        for seq in sequences:
            for ch in seq:
                if ch not in vocab:
                    vocab[ch] = idx
                    idx += 1
        return vocab

    def __len__(self):
        return len(self.inputs)

    def encode_seq(self, seq, vocab):
        return [vocab['<sos>']] + [vocab[ch] for ch in seq] + [vocab['<eos>']]

    def __getitem__(self, idx):
        src_seq = self.encode_seq(self.inputs[idx], self.src_vocab)
        tgt_seq = self.encode_seq(self.targets[idx], self.tgt_vocab)
        return torch.tensor(src_seq), torch.tensor(tgt_seq)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(seq) for seq in src_batch]
    tgt_lens = [len(seq) for seq in tgt_batch]

    src_pad = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    return src_pad, tgt_pad, src_lens, tgt_lens
