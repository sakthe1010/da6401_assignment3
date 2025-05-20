import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from dataset import TransliterationDataset, collate_fn
from model import Encoder, AttentionDecoder, Seq2SeqAttention
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans Tamil'

# Config (BEST from W&B sweep or train_attention.py)
CONFIG = {
    'batch_size': 64,
    'emb_dim': 64,
    'hidden_dim': 256,
    'attn_dim': 128,
    'enc_layers': 1,
    'dec_layers': 1,
    'cell_type': 'LSTM',
    'dropout': 0.2,
    'lr': 0.001,
    'epochs': 1
}
def visualize_attention_heatmaps(attn_weights, src_batch, pred_batch, idx2src, idx2tgt, save_dir="heatmaps"):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(len(attn_weights), 9)):
        src_str = clean_sequence(src_batch[i].tolist(), idx2src)
        pred_str = clean_sequence(pred_batch[i].tolist(), idx2tgt)

        attn = attn_weights[i][:len(pred_str), :len(src_str)].cpu().numpy()

        plt.figure(figsize=(6, 4))
        sns.heatmap(attn, xticklabels=list(src_str), yticklabels=list(pred_str),
                    cmap='viridis', cbar=True, linewidths=0.5)
        plt.xlabel("Input (Romanized)")
        plt.ylabel("Output (Tamil)")
        plt.title(f"Attention Heatmap {i+1}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/heatmap_{i+1}.png")
        plt.close()
def clean_sequence(seq, idx2char):
    result = []
    for i in seq:
        if i in [0, 1]: continue  # pad, sos
        if i == 2: break  # eos
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
    for src, tgt, src_lens, tgt_lens in tqdm(loader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output, _ = model(src, src_lens, tgt)
        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

def evaluate(model, loader, criterion, device, idx2src, idx2tgt):
    model.eval()
    total_loss = 0
    total_acc = 0
    predictions = []
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in loader:
            src, tgt = src.to(device), tgt.to(device)
            output, attn = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            acc = sequence_accuracy(output[:, 1:], tgt[:, 1:])
            total_loss += loss.item()
            total_acc += acc
            preds = output.argmax(dim=-1)
            for i in range(len(src)):
                src_str = clean_sequence(src[i].tolist(), idx2src)
                tgt_str = clean_sequence(tgt[i].tolist(), idx2tgt)
                pred_str = clean_sequence(preds[i].tolist(), idx2tgt)
                predictions.append((src_str, tgt_str, pred_str))
    return total_loss / len(loader), total_acc / len(loader), predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIG

    # Dataset
    train_set = TransliterationDataset("dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv")
    val_set = TransliterationDataset("dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv",
                                     src_vocab=train_set.src_vocab, tgt_vocab=train_set.tgt_vocab)
    test_set = TransliterationDataset("dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.test.tsv",
                                      src_vocab=train_set.src_vocab, tgt_vocab=train_set.tgt_vocab)

    # Combine train + val for final training
    full_train_set = torch.utils.data.ConcatDataset([train_set, val_set])

    train_loader = DataLoader(full_train_set, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], collate_fn=collate_fn)

    # Model
    encoder = Encoder(len(train_set.src_vocab), cfg['emb_dim'], cfg['hidden_dim'],
                      cfg['enc_layers'], cfg['cell_type'], cfg['dropout'])
    decoder = AttentionDecoder(len(train_set.tgt_vocab), cfg['emb_dim'], cfg['hidden_dim'],
                               cfg['dec_layers'], cfg['cell_type'], cfg['dropout'], cfg['attn_dim'])
    model = Seq2SeqAttention(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train
    for epoch in range(cfg['epochs']):
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    # Evaluate
    test_loss, test_acc, test_preds = evaluate(model, test_loader, criterion, device,
                                               train_set.src_idx2char, train_set.tgt_idx2char)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")

    # Save predictions
    os.makedirs("predictions_attention", exist_ok=True)
    pd.DataFrame(test_preds, columns=["Input", "Target", "Predicted"]).to_csv(
        "predictions_attention/predictions_attention.tsv", sep="\t", index=False)
    print("📄 Predictions saved to predictions_attention.tsv")

    # For attention heatmaps, take one batch again
    src, tgt, src_lens, tgt_lens = next(iter(test_loader))
    src, tgt = src.to(device), tgt.to(device)
    with torch.no_grad():
        output, attn_weights = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
        pred_tokens = output.argmax(dim=-1)

    visualize_attention_heatmaps(attn_weights, src, pred_tokens, train_set.src_idx2char, train_set.tgt_idx2char)
    print("📊 Saved attention heatmaps to heatmaps/")

if __name__ == "__main__":
    main()
