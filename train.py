# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2Seq
from dataset import TransliterationDataset, collate_fn
from tqdm import tqdm
import wandb


def train(model, iterator, optimizer, criterion, trg_pad_idx, clip=1.0):
    model.train()
    epoch_loss = 0

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

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, trg_pad_idx):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(model.device), trg.to(model.device)
            output = model(src, trg, teacher_forcing_ratio=0)  # no teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def main():
    wandb.init(
        project="transliteration-sweep-demo",
        config={
            "batch_size": 32,
            "emb_dim": 64,
            "hid_dim": 128,
            "n_layers": 1,
            "cell_type": "gru",
            "dropout": 0.2,
            "lr": 0.001,
            "epochs": 10
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths for tamil(ta)
    train_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv"
    valid_path = "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv"

    # Load dataset
    train_data = TransliterationDataset(train_path)
    valid_data = TransliterationDataset(valid_path)

    src_pad_idx, trg_pad_idx = train_data.get_pad_idx()
    input_dim, output_dim = train_data.get_vocab_sizes()

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_iter = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(input_dim, config.emb_dim, config.hid_dim, config.n_layers, config.cell_type, config.dropout)
    decoder = Decoder(output_dim, config.emb_dim, config.hid_dim, config.n_layers, config.cell_type, config.dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    best_valid_loss = float("inf")

    for epoch in range(config.epochs):
        train_loss = train(model, train_iter, optimizer, criterion, trg_pad_idx)
        valid_loss = evaluate(model, valid_iter, criterion, trg_pad_idx)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        })

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_seq2seq.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
