import torch
import torch.nn as nn

def get_rnn(cell_type):
    cell_type = cell_type.lower()
    if cell_type == 'lstm':
        return nn.LSTM
    elif cell_type == 'gru':
        return nn.GRU
    else:
        return nn.RNN

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, cell='gru', dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = get_rnn(cell)(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, cell='gru', dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = get_rnn(cell)(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
    
def beam_decode(self, src, sos_idx, eos_idx, max_len=30, beam_size=3):
    """
    Beam search decoding for a single example (batch size = 1).
    """
    self.encoder.eval()
    self.decoder.eval()

    with torch.no_grad():
        src = src.unsqueeze(1)  # [src_len, 1]
        encoder_hidden = self.encoder(src)
        beams = [( [sos_idx], 0.0, encoder_hidden )]  # (token_seq, score, hidden)

        completed = []

        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden in beams:
                last_token = torch.tensor([seq[-1]], device=self.device)

                output, hidden = self.decoder(last_token, hidden)
                log_probs = torch.log_softmax(output, dim=-1).squeeze(0)

                topk = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    next_token = topk.indices[i].item()
                    next_score = score + topk.values[i].item()
                    new_seq = seq + [next_token]

                    if next_token == eos_idx:
                        completed.append((new_seq, next_score))
                    else:
                        new_beams.append((new_seq, next_score, hidden))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if not completed:
            completed = beams
        best_seq = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
        return best_seq

