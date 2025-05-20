import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
from dataset import TransliterationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

# Fresh run project
wandb.init(project="ASSIGNMENT_3", config={
    'batch_size': 64,
    'emb_dim': 128,
    'hidden_dim': 256,
    'enc_layers': 1,
    'dec_layers': 1,
    'cell_type': 'LSTM',
    'dropout': 0.2,
    'lr': 0.001,
    'epochs': 30,
})

def clean_sequence(seq, idx2char):
    result = []
    for i in seq:
        if i in [0, 1]:
            continue
        if i == 2:
            break
        result.append(idx2char.get(i, ""))
    return ''.join(result)

def sequence_accuracy(preds, targets, pad_idx=0):
    pred_tokens = preds.argmax(dim=-1)
    correct = 0
    total = 0
    for pred_seq, tgt_seq in zip(pred_tokens, targets):
        pred_seq = [x for x in pred_seq.tolist() if x not in [pad_idx, 1]]
        tgt_seq = [x for x in tgt_seq.tolist() if x not in [pad_idx, 1]]
        if 2 in pred_seq:
            pred_seq = pred_seq[:pred_seq.index(2)]
        if 2 in tgt_seq:
            tgt_seq = tgt_seq[:tgt_seq.index(2)]
        if pred_seq == tgt_seq:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    pbar = tqdm(loader, desc="Training")
    for src, tgt, src_lens, tgt_lens in pbar:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_lens, tgt)
        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        acc = sequence_accuracy(output[:, 1:], tgt[:, 1:])
        total_loss += loss.item()
        total_acc += acc
        pbar.set_postfix(loss=loss.item(), acc=acc)
    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, criterion, device, src_idx2char, tgt_idx2char):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_preds = []
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            acc = sequence_accuracy(output[:, 1:], tgt[:, 1:])
            total_loss += loss.item()
            total_acc += acc
            pred_tokens = output.argmax(dim=-1)
            for i in range(len(src)):
                src_str = clean_sequence(src[i].tolist(), src_idx2char)
                tgt_str = clean_sequence(tgt[i].tolist(), tgt_idx2char)
                pred_str = clean_sequence(pred_tokens[i].tolist(), tgt_idx2char)
                all_preds.append((src_str, tgt_str, pred_str))
    return total_loss / len(loader), total_acc / len(loader), all_preds

def save_best_predictions(predictions):
    df = pd.DataFrame(predictions, columns=["Input", "Target", "Predicted"])
    df.to_csv("best_predictions.tsv", sep="\t", index=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = wandb.config

    train_set = TransliterationDataset("dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv")
    val_set = TransliterationDataset("dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv",
                                     src_vocab=train_set.src_vocab, tgt_vocab=train_set.tgt_vocab)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, collate_fn=collate_fn)

    encoder = Encoder(len(train_set.src_vocab), config.emb_dim, config.hidden_dim, config.enc_layers, config.cell_type, config.dropout)
    decoder = Decoder(len(train_set.tgt_vocab), config.emb_dim, config.hidden_dim, config.dec_layers, config.cell_type, config.dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_acc = 0.0
    best_predictions = []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds = evaluate(model, val_loader, criterion, device, train_set.src_idx2char, train_set.tgt_idx2char)

        print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            best_predictions = val_preds
            print(f"✅ New best model saved (val_acc={val_acc:.4f})")

    save_best_predictions(best_predictions)
    print(f"\n🎯 Best validation accuracy: {best_val_acc:.4f}")
    print("✔️ Saved best model and predictions to disk.")

if __name__ == "__main__":
    main()
