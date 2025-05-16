import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2Seq
from dataset import TransliterationDataset, collate_fn
from tqdm import tqdm

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
            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def main():
    BATCH_SIZE = 64
    EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 1
    CELL_TYPE = 'gru'  # 'lstm' or 'rnn'
    DROPOUT = 0.2
    N_EPOCHS = 15
    LR = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_path = "path/to/ta.translit.sampled.train.tsv"  # ⬅️ update path
    valid_path = "path/to/ta.translit.sampled.dev.tsv"    # ⬅️ update path

    train_data = TransliterationDataset(train_path)
    valid_data = TransliterationDataset(valid_path)

    src_pad_idx, trg_pad_idx = train_data.get_pad_idx()
    input_dim, output_dim = train_data.get_vocab_sizes()

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Instantiate model
    encoder = Encoder(input_dim, EMB_DIM, HID_DIM, N_LAYERS, cell=CELL_TYPE, dropout=DROPOUT)
    decoder = Decoder(output_dim, EMB_DIM, HID_DIM, N_LAYERS, cell=CELL_TYPE, dropout=DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iter, optimizer, criterion, trg_pad_idx)
        valid_loss = evaluate(model, valid_iter, criterion, trg_pad_idx)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_seq2seq.pt")

if __name__ == "__main__":
    main()
