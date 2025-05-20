import torch
from model import Encoder, Decoder, Seq2Seq, AttentionDecoder, Seq2SeqAttention
from dataset import get_vocabs
from connectivity import collect_connectivity
from viz import plot_connectivity

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Hyperparameters (MUST MATCH your training) ----------
EMB_DIM     = 256
HID_DIM     = 512
NLAYERS     = 2
CELL_TYPE   = "GRU"     # "LSTM" or "GRU"
DROPOUT     = 0.2
ATTN        = False     # Set to True if you're using attention

# ---------- 2. Rebuild model architecture ----------
src_c2i, tgt_c2i, src_i2c, tgt_i2c = get_vocabs()
INPUT_DIM  = len(src_c2i)
OUTPUT_DIM = len(tgt_c2i)

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NLAYERS, CELL_TYPE, DROPOUT)
if ATTN:
    decoder = AttentionDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NLAYERS, CELL_TYPE, DROPOUT, attn_dim=HID_DIM)
    model = Seq2SeqAttention(encoder, decoder, device)
else:
    decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NLAYERS, CELL_TYPE, DROPOUT)
    model = Seq2Seq(encoder, decoder, device)

model.to(device)

# ---------- 3. Load weights ----------
ckpt = torch.load("best.ckpt", map_location=device)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
model.load_state_dict(ckpt, strict=False)
model.eval()

# ---------- 4. Input example (replace with actual words) ----------
src_word = "grammar"
tgt_word = "கிராமர்"  # Use correct Tamil ground truth here

src_tensor = torch.tensor([src_c2i[c] for c in src_word])
tgt_tensor = torch.tensor([tgt_c2i[c] for c in tgt_word])
sos_idx = tgt_c2i['<sos>']

# ---------- 5. Run saliency + plot ----------
infl, xlab, ylab = collect_connectivity(model, src_tensor, tgt_tensor,
                                        src_i2c, tgt_i2c, sos_idx=sos_idx, device=device)
plot_connectivity(infl, xlab, ylab)
