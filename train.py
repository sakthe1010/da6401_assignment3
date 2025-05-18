import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2Seq
from dataset import TransliterationDataset, collate_fn
from tqdm import tqdm
import wandb

def decode_seq(seq_tensor, idx2char):
    chars = []
    for idx in seq_tensor:
        ch = idx2char.get(idx.item(), "")
        if ch == "<eos>": break
        if ch not in ["<pad>", "<sos>"]:
            chars.append(ch)
    return "".join(chars)

def exact_match(preds, targets):
    correct = sum([p == t for p, t in zip(preds, targets)])
    return correct / len(preds) if preds else 0.0

def train(model, iterator, optimizer, criterion, trg_pad_idx, clip=1.0):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0

    for src, trg in tqdm(iterator):
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        pred = output.argmax(1)
        mask = trg != trg_pad_idx
        correct = (pred == trg) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item()

    acc = total_correct / total_tokens
    return total_loss / len(iterator), acc

def evaluate(model, iterator, criterion, trg_pad_idx, trg_idx2char, src_idx2char, beam_size=1, sos_idx=1, eos_idx=2, log_examples=5):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    pred_strs, tgt_strs, input_strs = [], [], []

    with torch.no_grad():
        for batch_idx, (src, trg) in enumerate(iterator):
            src, trg = src.to(model.device), trg.to(model.device)

            if beam_size == 1:
                # Greedy decoding for efficient evaluation
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg_flat = trg[1:].view(-1)

                loss = criterion(output, trg_flat)
                total_loss += loss.item()

                pred = output.argmax(1)
                mask = trg_flat != trg_pad_idx
                correct = (pred == trg_flat) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

                if batch_idx == 0:
                    for i in range(min(log_examples, src.shape[1])):
                        pred_seq = pred.view(-1, src.shape[1])[:, i]
                        trg_seq = trg_flat.view(-1, src.shape[1])[:, i]
                        pred_str = decode_seq(pred_seq, trg_idx2char)
                        tgt_str = decode_seq(trg_seq, trg_idx2char)
                        inp_str = decode_seq(src[:, i], src_idx2char)
                        print(f"[EXAMPLE {i}] INPUT: {inp_str} | PRED: {pred_str} | TRUE: {tgt_str}")

            else:
                # Beam decoding for accurate evaluation (more computationally expensive)
                for i in range(src.shape[1]):
                    src_seq = src[:, i]
                    trg_seq = trg[:, i]

                    pred_idx_seq = model.beam_decode(src_seq, sos_idx, eos_idx, beam_size=beam_size)
                    pred_str = decode_seq(torch.tensor(pred_idx_seq), trg_idx2char)
                    tgt_str = decode_seq(trg_seq, trg_idx2char)
                    inp_str = decode_seq(src_seq, src_idx2char)

                    pred_strs.append(pred_str)
                    tgt_strs.append(tgt_str)
                    input_strs.append(inp_str)

                    if batch_idx == 0 and i < log_examples:
                        print(f"[EXAMPLE {i}] INPUT: {inp_str} | PRED: {pred_str} | TRUE: {tgt_str}")

        # Calculate accuracy
        if beam_size == 1:
            acc = total_correct / total_tokens
        else:
            acc = exact_match(pred_strs, tgt_strs)
            wandb.log({
                "sample_predictions": [f"IN: {inp} | PRED: {p} | TRUE: {t}" 
                                       for inp, p, t in zip(input_strs[:log_examples], pred_strs[:log_examples], tgt_strs[:log_examples])]
            })

    avg_loss = total_loss / len(iterator)
    return avg_loss, acc


def main():
    wandb.init(
        project="transliteration-sweep",
        config={
            "batch_size": 32,
            "emb_dim": 64,
            "hid_dim": 128,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "cell_type": "gru",
            "dropout": 0.3,
            "lr": 0.0005,
            "beam_size": 3,
            "epochs": 5
        }
    )
    config = wandb.config

    run_name = f"emb{config.emb_dim}-hid{config.hid_dim}-enc{config.n_encoder_layers}-dec{config.n_decoder_layers}-{config.cell_type}-drop{config.dropout}-bs{config.batch_size}-beam{config.beam_size}-lr{config.lr}"
    wandb.run.name = run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
    valid_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"

    train_data = TransliterationDataset(train_path)
    valid_data = TransliterationDataset(valid_path)

    src_pad_idx, trg_pad_idx = train_data.get_pad_idx()
    input_dim, output_dim = train_data.get_vocab_sizes()

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    dropout = config.dropout if config.n_encoder_layers > 1 or config.n_decoder_layers > 1 else 0.0

    encoder = Encoder(input_dim, config.emb_dim, config.hid_dim, config.n_encoder_layers, config.cell_type, dropout)
    decoder = Decoder(output_dim, config.emb_dim, config.hid_dim, config.n_decoder_layers, config.cell_type, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, trg_pad_idx)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, trg_pad_idx,
                                     valid_data.trg_idx2char, valid_data.src_idx2char,  beam_size=1)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": val_loss,
            "valid_acc": val_acc
        })

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_seq2seq.pt")

    wandb.finish()

if __name__ == "__main__":
    main()
