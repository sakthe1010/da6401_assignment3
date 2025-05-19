import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset import read_data, build_vocab, TransliterationDataset
from model import Encoder, Decoder, Seq2Seq

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = "dakshina_dataset_v1.0/ta/lexicons"
TRAIN_FILE= os.path.join(DATA_DIR, "ta.translit.sampled.train.tsv")
DEV_FILE  = os.path.join(DATA_DIR, "ta.translit.sampled.dev.tsv")
MAX_LEN   = 20   # ← set based on 95th percentile

# ----------------------------
# ACCURACY METRICS
# ----------------------------
def sequence_accuracy(pred, trg, pad_idx):
    pred_ids = pred.argmax(-1)
    mask     = trg != pad_idx
    equal    = (pred_ids == trg) | ~mask
    seq_corr = equal.all(dim=1).float()
    return seq_corr.mean().item()

def char_accuracy(pred, trg, pad_idx):
    pred_ids = pred.argmax(-1)
    mask     = trg != pad_idx
    correct  = (pred_ids == trg) & mask
    return correct.sum().item() / mask.sum().item()

# ----------------------------
# TRAIN / VALIDATION
# ----------------------------
def train_epoch(model, loader, optimizer, criterion, pad_idx):
    model.train()
    tot_loss = tot_seq_acc = tot_char_acc = 0.0
    pbar = tqdm(loader, desc="Training", unit="batch")
    for src, trg in pbar:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)
        loss   = criterion(
            output[:,1:].reshape(-1, output.size(-1)),
            trg   [:,1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()

        seq_acc  = sequence_accuracy(output, trg, pad_idx)
        char_acc = char_accuracy  (output, trg, pad_idx)

        tot_loss     += loss.item()
        tot_seq_acc  += seq_acc
        tot_char_acc += char_acc

        pbar.set_postfix({
            "loss":     tot_loss / (pbar.n+1),
            "seq_acc":  tot_seq_acc / (pbar.n+1),
            "char_acc": tot_char_acc / (pbar.n+1),
        })

    n = len(loader)
    return tot_loss/n, tot_seq_acc/n, tot_char_acc/n

def eval_epoch(model, loader, criterion, pad_idx):
    model.eval()
    tot_seq_acc = tot_char_acc = 0.0
    pbar = tqdm(loader, desc="Validating", unit="batch", leave=False)
    with torch.no_grad():
        for src, trg in pbar:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            seq_acc  = sequence_accuracy(output, trg, pad_idx)
            char_acc = char_accuracy  (output, trg, pad_idx)

            tot_seq_acc  += seq_acc
            tot_char_acc += char_acc

            pbar.set_postfix({
                "seq_acc":  tot_seq_acc  / (pbar.n+1),
                "char_acc": tot_char_acc / (pbar.n+1),
            })

    n = len(loader)
    return tot_seq_acc/n, tot_char_acc/n

# ----------------------------
# MAIN
# ----------------------------
def main(config=None):
    with wandb.init(
        project="transliteration-sweep",
        config=config or {
            "batch_size":       32,
            "emb_dim":          64,
            "hid_dim":          128,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "cell_type":        "GRU",
            "dropout":          0.3,
            "lr":               0.0005,
            "beam_size":        3,
            "epochs":           5
        }
    ):
        cfg = wandb.config
        run_name = (
            f"emb{cfg.emb_dim}-hid{cfg.hid_dim}"
            f"-enc{cfg.n_encoder_layers}-dec{cfg.n_decoder_layers}"
            f"-{cfg.cell_type}-drop{cfg.dropout}"
            f"-bs{cfg.batch_size}-beam{cfg.beam_size}-lr{cfg.lr}"
        )
        wandb.run.name = run_name

        # Load data
        train_pairs = read_data(TRAIN_FILE)
        dev_pairs   = read_data(DEV_FILE)
        char2idx, _ = build_vocab(train_pairs + dev_pairs)
        pad_idx     = char2idx['<pad>']

        if not train_pairs:
            print("[FATAL] No training data found. Check paths/format.")
            return

        train_ds = TransliterationDataset(train_pairs, char2idx, char2idx, MAX_LEN, MAX_LEN)
        dev_ds   = TransliterationDataset(dev_pairs,   char2idx, char2idx, MAX_LEN, MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size)

        # Build model
        encoder = Encoder(
            input_dim=len(char2idx),
            emb_dim=cfg.emb_dim,
            hid_dim=cfg.hid_dim,
            n_layers=cfg.n_encoder_layers,
            cell_type=cfg.cell_type,
            dropout=cfg.dropout
        )
        decoder = Decoder(
            output_dim=len(char2idx),
            emb_dim=cfg.emb_dim,
            hid_dim=cfg.hid_dim,
            n_layers=cfg.n_decoder_layers,
            cell_type=cfg.cell_type,
            dropout=cfg.dropout
        )
        model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        best_val_seq_acc = 0.0
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_seq_acc, train_char_acc = train_epoch(
                model, train_loader, optimizer, criterion, pad_idx
            )
            val_seq_acc, val_char_acc = eval_epoch(
                model, dev_loader, criterion, pad_idx
            )

            wandb.log({
                "train_loss":    train_loss,
                "train_seq_acc": train_seq_acc,
                "train_char_acc":train_char_acc,
                "val_seq_acc":   val_seq_acc,
                "val_char_acc":  val_char_acc,
                "epoch":         epoch
            })
            print(
                f"Epoch {epoch} | "
                f"Tr Loss: {train_loss:.4f} | Tr SeqAcc: {train_seq_acc:.4f} | Tr CharAcc: {train_char_acc:.4f} | "
                f"Val SeqAcc: {val_seq_acc:.4f} | Val CharAcc: {val_char_acc:.4f}"
            )

            if val_seq_acc > best_val_seq_acc:
                best_val_seq_acc = val_seq_acc
                torch.save(
                    model.state_dict(),
                    f"best_model_{wandb.run.name}.pt"
                )
                print(f"✅ Saved best model at epoch {epoch} with Val SeqAcc = {val_seq_acc:.4f}")

if __name__ == "__main__":
    main()
