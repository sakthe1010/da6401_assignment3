#!/usr/bin/env python3
# visualize_lstm_vs_gru_manual.py

import argparse
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import Encoder, Decoder, AttentionDecoder, Seq2Seq, Seq2SeqAttention
from dataset import TransliterationDataset, get_vocabs

import plotly.io as pio
pio.renderers.default = "browser"

# ─── Hard-coded hyper-parameters ────────────────────────────────────────────
GRU_PARAMS = {
    "emb_dim":    32,
    "hidden_dim":128,
    "enc_layers": 3,
    "dec_layers": 3,
    "dropout":   0.3
}

LSTM_PARAMS = {
    "emb_dim":    64,
    "hidden_dim": 256,
    "enc_layers": 1,
    "dec_layers": 1,
    "dropout":   0.2,
    "attn_dim":  128
}
# ─────────────────────────────────────────────────────────────────────────────

def load_vanilla_gru(path, device):
    p = GRU_PARAMS
    input_dim  = len(src_char2idx)
    output_dim = len(tgt_char2idx)
    enc = Encoder(input_dim, p["emb_dim"], p["hidden_dim"], p["enc_layers"], "GRU", p["dropout"])
    dec = Decoder(output_dim, p["emb_dim"], p["hidden_dim"], p["dec_layers"], "GRU", p["dropout"])
    model = Seq2Seq(enc, dec, device).to(device)
    sd = torch.load(path, map_location=device)
    # if you wrapped state dict in 'model_state_dict':
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd)
    model.eval()
    return model

def load_lstm_attention(path, device):
    p = LSTM_PARAMS
    input_dim  = len(src_char2idx)
    output_dim = len(tgt_char2idx)
    enc = Encoder(input_dim, p["emb_dim"], p["hidden_dim"], p["enc_layers"], "LSTM", p["dropout"])
    dec = AttentionDecoder(output_dim, p["emb_dim"], p["hidden_dim"], p["dec_layers"], "LSTM", p["dropout"], p["attn_dim"])
    model = Seq2SeqAttention(enc, dec, device).to(device)
    sd = torch.load(path, map_location=device)
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd)
    model.eval()
    return model

def make_seq(word, mapping):
    ids = [mapping["<sos>"]] + [mapping[c] for c in word] + [mapping["<eos>"]]
    return torch.LongTensor([ids])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True, help="Ta train TSV")
    parser.add_argument("--lstm_ckpt", required=True)
    parser.add_argument("--gru_ckpt",  required=True)
    parser.add_argument("--src",       required=True)
    parser.add_argument("--tgt",       required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load vocab
    TransliterationDataset(args.dataset)
    global src_char2idx, tgt_char2idx, src_idx2char, tgt_idx2char
    src_char2idx, tgt_char2idx, src_idx2char, tgt_idx2char = get_vocabs()

    # load both models
    gru_model  = load_vanilla_gru( args.gru_ckpt,  device)
    lstm_model = load_lstm_attention(args.lstm_ckpt, device)

    # build input/output tensors
    src_t = make_seq(args.src, src_char2idx).to(device)
    tgt_t = make_seq(args.tgt, tgt_char2idx).to(device)
    src_len = [src_t.size(1)]

    # forward pass (full teacher forcing)
    with torch.no_grad():
        # vanilla GRU returns only the outputs
        gru_logits = gru_model(src_t, src_len, tgt_t, teacher_forcing_ratio=1.0)
        # LSTM+attention returns (outputs, attn_weights)
        lstm_logits, attn = lstm_model(src_t, src_len, tgt_t, teacher_forcing_ratio=1.0)

    # print GRU output
    pred_ids = gru_logits.argmax(-1).squeeze(0).cpu().tolist()
    pred_chars = [tgt_idx2char[i] for i in pred_ids][1:]  # drop <sos>
    print(f"\nGRU vanilla output: {''.join(pred_chars)}\n")

    # attention matrix
    attn = attn.squeeze(0).cpu().numpy()  # shape: (T_out, T_in)

    # labels
    src_seq = [src_idx2char[i] for i in src_t.squeeze(0).cpu().tolist()]
    tgt_seq = [tgt_idx2char[i] for i in tgt_t.squeeze(0).cpu().tolist()][1:]

    # Plotly figure: heatmap + step-wise bar + slider
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type":"heatmap"}],[{"type":"xy"}]],
        row_heights=[0.7,0.3],
        subplot_titles=["LSTM + Attention","Attention at Step 0"]
    )

    fig.add_trace(go.Heatmap(
        z=attn, x=src_seq, y=tgt_seq,
        colorscale="Blues",
        hovertemplate="out %{y} ← in %{x}<br>wt %{z:.3f}"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=src_seq, y=attn[0],
        name=f"step 0 → {tgt_seq[0]}"
    ), row=2, col=1)

    # frames & slider
    frames = [go.Frame(
        data=[{"y": attn[i]}],
        name=str(i),
        traces=[1]
    ) for i in range(len(tgt_seq))]

    steps = [dict(
        method="animate",
        args=[[str(i)],{"mode":"immediate","frame":{"duration":300,"redraw":True},"transition":{"duration":0}}],
        label=f"{i}:{tgt_seq[i]}"
    ) for i in range(len(tgt_seq))]

    fig.frames = frames
    fig.update_layout(
        sliders=[dict(active=0,pad={"t":50},steps=steps)],
        height=800, width=700,
        title="LSTM Attention (GRU vanilla above)"
    )

    fig.write_html("attention_viz.html", auto_open=True)

if __name__=="__main__":
    main()
