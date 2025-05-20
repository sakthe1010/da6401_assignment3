import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Shared Encoder ----------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = getattr(nn, cell_type)(
            emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

# ---------------------- Vanilla Decoder & Seq2Seq ----------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = getattr(nn, cell_type)(
            emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
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

        encoder_outputs, hidden = self.encoder(src, src_lens)

        def adjust_hidden(h, target_layers):
            if h.size(0) > target_layers:
                return h[:target_layers]
            elif h.size(0) < target_layers:
                reps = target_layers // h.size(0)
                return h.repeat(reps, 1, 1)
            return h

        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            h = adjust_hidden(h, self.decoder.rnn.num_layers)
            c = adjust_hidden(c, self.decoder.rnn.num_layers)
            hidden = (h, c)
        else:
            hidden = adjust_hidden(hidden, self.decoder.rnn.num_layers)

        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

# ---------------------- Attention Decoder & Seq2SeqAttention ----------------------
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        if decoder_hidden.dim() == 3:  # LSTM: [layers, batch, hidden]
            decoder_hidden = decoder_hidden[-1]  # last layer
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attn_scores = self.v(energy).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)

        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, cell_type, dropout, attn_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = getattr(nn, cell_type)(
            emb_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = Attention(hidden_dim, hidden_dim, attn_dim)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        context, attn_weights = self.attention(hidden[0] if isinstance(hidden, tuple) else hidden, encoder_outputs, mask)
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(torch.cat((output.squeeze(1), context), dim=1))
        return prediction, hidden, attn_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        attn_weights_all = []

        encoder_outputs, hidden = self.encoder(src, src_lens)

        def adjust_hidden(h, target_layers):
            if h.size(0) > target_layers:
                return h[:target_layers]
            elif h.size(0) < target_layers:
                reps = target_layers // h.size(0)
                return h.repeat(reps, 1, 1)
            return h

        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            h = adjust_hidden(h, self.decoder.rnn.num_layers)
            c = adjust_hidden(c, self.decoder.rnn.num_layers)
            hidden = (h, c)
        else:
            hidden = adjust_hidden(hidden, self.decoder.rnn.num_layers)

        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            attn_weights_all.append(attn_weights.unsqueeze(1))
            top1 = output.argmax(1)
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        attn_weights_all = torch.cat(attn_weights_all, dim=1)  # [batch, tgt_len - 1, src_len]
        return outputs, attn_weights_all
