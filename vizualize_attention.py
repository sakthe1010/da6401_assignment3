#!/usr/bin/env python3
# visualize_attention.py

import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets

from model import Encoder, AttentionDecoder, Seq2SeqAttention

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) your vocab mappings: char→idx and idx→char
#    Replace these with your actual mappings (e.g. load from pickle/json)
src_stoi = {
    '<sos>':0, '<eos>':1,
    **{c:i+2 for i,c in enumerate(list("abcdefghijklmnopqrstuvwxyz"))}
}
# inverse mapping for output tokens
tgt_itos = {v:k for k,v in src_stoi.items()}

# 2) checkpoint paths
LSTM_CHECKPOINT = 'best_attention_model.pt'
GRU_CHECKPOINT  = 'best_vanilla_model.pt'

# 3) which model to visualize?  set to 'LSTM' or 'GRU'
MODEL_TYPE = 'LSTM'

# 4) your test input
SRC_WORD = "context"
# ────────────────────────────────────────────────────────────────────────────────

def build_model(cell_type, ckpt_path):
    # must match your training hyperparams!
    INPUT_DIM  = len(src_stoi)
    OUTPUT_DIM = len(tgt_itos)
    EMB_DIM    = 256
    HID_DIM    = 512
    N_LAYERS   = 2
    DROPOUT    = 0.1
    ATTN_DIM   = 64

    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, cell_type, DROPOUT)
    dec = AttentionDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, cell_type, DROPOUT, ATTN_DIM)
    model = Seq2SeqAttention(enc, dec, DEVICE).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def tensor_from_word(word, stoi, device):
    ids = [stoi['<sos>']] + [stoi[c] for c in word] + [stoi['<eos>']]
    return torch.LongTensor(ids).unsqueeze(0).to(device)  # (1, T_in)

def get_attention_matrix(model, src_tensor):
    # run with teacher forcing off (so it actually decodes)
    with torch.no_grad():
        # we pass src as both src & tgt so we step one by one
        logits, attn = model(src_tensor, [src_tensor.size(1)]*1, src_tensor, teacher_forcing_ratio=0.0)
    # attn: (batch=1, T_out-1, T_in)
    return attn.squeeze(0).cpu().numpy()

def plot_heatmap(attn, src_chars, tgt_chars):
    fig = px.imshow(
        attn,
        x=src_chars,
        y=tgt_chars,
        labels={'x':'Input char','y':'Output step'},
        aspect='auto',
        color_continuous_scale='Blues'
    )
    fig.update_traces(
        hovertemplate="decode %{y} ← src %{x}<br>weight %{z:.3f}"
    )
    fig.update_layout(title="Full Attention Heatmap")
    fig.show()

def interactive_bar(attn, src_chars, tgt_chars):
    def plot_step(i):
        weights = attn[i]  # shape (T_in,)
        fig = go.Figure(go.Bar(
            x=src_chars, y=weights,
            hovertemplate="src %{x}: %{y:.3f}<extra></extra>"
        ))
        fig.update_layout(
          title=f"Attention at output step {i} ({tgt_chars[i]})",
          xaxis_title="input char", yaxis_title="weight",
          width=700, height=350
        )
        fig.show()

    widgets.interact(
        plot_step,
        i=widgets.IntSlider(min=0, max=attn.shape[0]-1, step=1, value=0)
    )

def main():
    # pick model
    if MODEL_TYPE.upper() == 'LSTM':
        ckpt = LSTM_CHECKPOINT
        cell = 'LSTM'
    else:
        ckpt = GRU_CHECKPOINT
        cell = 'GRU'

    model = build_model(cell, ckpt)
    src_tensor = tensor_from_word(SRC_WORD, src_stoi, DEVICE)

    # get raw attention matrix
    attn = get_attention_matrix(model, src_tensor)

    # axis labels
    src_chars = ['<sos>'] + list(SRC_WORD) + ['<eos>']
    # pretend your model predicted the same number of steps
    tgt_chars = [f"step{t}" for t in range(attn.shape[0])]

    # 1) full heatmap
    plot_heatmap(attn, src_chars, tgt_chars)

    # 2) per-step bar + slider
    interactive_bar(attn, src_chars, tgt_chars)

if __name__ == "__main__":
    main()
