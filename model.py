import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cls = getattr(nn, cell_type.upper())
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = getattr(nn, cell_type.upper())
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        # input: [B], turn into [B,1]
        embedded = self.embedding(input.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        B, T = trg.shape
        V = self.decoder.fc_out.out_features
        outputs = torch.zeros(B, T, V).to(self.device)

        hidden = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, T):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

    def beam_search(self, src, sos_idx, eos_idx, beam_size, max_len):
        hidden = self.encoder(src.unsqueeze(0))
        beams = [(torch.tensor([sos_idx], device=self.device), 0.0, hidden)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for seq, score, hidden in beams:
                if seq[-1].item() == eos_idx:
                    completed.append((seq, score))
                    continue
                output, hidden = self.decoder(seq[-1].view(1), hidden)
                log_probs = F.log_softmax(output, dim=1).squeeze(0)
                topk = torch.topk(log_probs, beam_size)
                for idx, lp in zip(topk.indices, topk.values):
                    candidates.append((torch.cat([seq, idx.view(1)]), score + lp.item(), hidden))
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        completed += beams
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        return completed[0][0]
