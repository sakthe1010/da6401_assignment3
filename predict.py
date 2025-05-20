import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TransliterationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq
from train import evaluate, clean_sequence  # reuse utility functions

def save_predictions(predictions, filename):
    df = pd.DataFrame(predictions, columns=["Input", "Target", "Predicted"])
    df.to_csv(filename, sep="\t", index=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint containing model + config + vocabs
    checkpoint = torch.load("assignment_3/best_model_full.pt")

    config = checkpoint["config"]
    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]

    src_idx2char = {i: ch for ch, i in src_vocab.items()}
    tgt_idx2char = {i: ch for ch, i in tgt_vocab.items()}

    # Load test set using vocab from training
    test_set = TransliterationDataset(
        "dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.test.tsv",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab
    )
    test_loader = DataLoader(test_set, batch_size=config['batch_size'],
                             collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # Build model
    encoder = Encoder(len(src_vocab), config['emb_dim'], config['hidden_dim'],
                      config['enc_layers'], config['cell_type'], config['dropout'])
    decoder = Decoder(len(tgt_vocab), config['emb_dim'], config['hidden_dim'],
                      config['dec_layers'], config['cell_type'], config['dropout'])
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load model weights
    print("Loading best model weights...")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy loss fn for interface
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Predict
    test_loss, test_acc, test_preds = evaluate(
        model, test_loader, criterion, device,
        src_idx2char, tgt_idx2char
    )

    print(f"\n🧪 Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    save_predictions(test_preds, "predictionss_vanilla/best_model_test_predictions.tsv")
    print("📄 Test predictions saved to: predictions_vanilla/test_predictions.tsv")

if __name__ == "__main__":
    main()
