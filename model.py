import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = getattr(nn, cell_type)(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
    
    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = getattr(nn, cell_type)(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        hidden = self.encoder(src, src_lens)
        # Adjust hidden state to match decoder layers if needed
        if isinstance(hidden, tuple):  # LSTM (hidden, cell)
            h, c = hidden
            dec_layers = self.decoder.rnn.num_layers
            if h.size(0) != dec_layers:
                h = h.repeat(dec_layers, 1, 1)
                c = c.repeat(dec_layers, 1, 1)
            hidden = (h, c)
        else:  # GRU or RNN
            if hidden.size(0) != self.decoder.rnn.num_layers:
                hidden = hidden.repeat(self.decoder.rnn.num_layers, 1, 1)
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
