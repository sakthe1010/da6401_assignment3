import torch
import wandb
import numpy as np
import plotly.graph_objects as go
from model import Encoder, Decoder, AttentionDecoder, Seq2Seq, Seq2SeqAttention
from dataset import get_vocabs
from connectivity import collect_connectivity

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_saliency(model_file, use_attention, cell_type, src_word, tgt_word):
    src_c2i, tgt_c2i, src_i2c, tgt_i2c = get_vocabs()
    INPUT_DIM  = len(src_c2i)
    OUTPUT_DIM = len(tgt_c2i)
    EMB_DIM, HID_DIM, NLAYERS, DROPOUT = 256, 512, 2, 0.2
    SOS_ID = tgt_c2i["<sos>"]

    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NLAYERS, cell_type, DROPOUT)
    if use_attention:
        decoder = AttentionDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NLAYERS, cell_type, DROPOUT, attn_dim=HID_DIM)
        model = Seq2SeqAttention(encoder, decoder, device)
    else:
        decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NLAYERS, cell_type, DROPOUT)
        model = Seq2Seq(encoder, decoder, device)

    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    model.to(device).eval()

    src_tensor = torch.tensor([src_c2i[c] for c in src_word])
    tgt_tensor = torch.tensor([tgt_c2i[c] for c in tgt_word])
    infl, xlab, ylab = collect_connectivity(model, src_tensor, tgt_tensor,
                                            src_i2c, tgt_i2c, sos_idx=SOS_ID, device=device)
    return infl, xlab, ylab

def plot_dual_connectivity(gru_infl, lstm_infl, x_chars, y_chars, run=None, title="LSTM vs GRU"):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=lstm_infl,
        x=x_chars,
        y=y_chars,
        colorscale='Viridis',
        showscale=False,
        hovertemplate="LSTM<br><b>%{y}</b> depends on <b>%{x}</b><extra></extra>",
    ))

    fig.add_trace(go.Heatmap(
        z=gru_infl,
        x=x_chars,
        y=y_chars,
        colorscale='Viridis',
        showscale=True,
        yaxis='y2',
        hovertemplate="GRU<br><b>%{y}</b> depends on <b>%{x}</b><extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=600,
        margin=dict(t=60, b=40),
        yaxis=dict(domain=[0.55, 1], title="LSTM"),
        yaxis2=dict(domain=[0, 0.45], title="GRU"),
    )

    if run:
        html = fig.to_html(include_plotlyjs='cdn')
        run.log({title: wandb.Html(html)})

    fig.show()


if __name__ == "__main__":
    wandb.init(project="ASSIGNMENT_3_CONNECTIVITY", name="LSTM vs GRU Comparison")

    # Input sentence and expected Tamil output
    src_word = "context the formal study of grammar is an important part of education"
    tgt_word = "காண்டெக்ஸ்ட் தி ஃபார்மல் ஸ்டடி ஆஃப் கிராமர் இஸ் அன இம்போர்டண்ட் பார்ட் ஆஃப் எடுகேஷன்"
    tgt_word = tgt_word[:len(src_word)]  # truncate to match length

    # Collect saliency
    lstm_infl, x_chars, y_chars = get_saliency(
        model_file="best_attention_model.pt",
        use_attention=True,
        cell_type="LSTM",
        src_word=src_word,
        tgt_word=tgt_word
    )

    gru_infl, _, _ = get_saliency(
        model_file="best_vanilla_model.pt",
        use_attention=False,
        cell_type="GRU",
        src_word=src_word,
        tgt_word=tgt_word
    )

    # Plot and log to W&B
    plot_dual_connectivity(gru_infl, lstm_infl, x_chars, y_chars, run=wandb.run)
    wandb.finish()
