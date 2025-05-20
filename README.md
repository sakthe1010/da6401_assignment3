# Tamil Transliteration with Seq2Seq (DA6401 Assignment 3)

This repository implements a neural transliteration system for converting Romanized Tamil script to native Tamil script using sequence-to-sequence models. It supports both **Vanilla Seq2Seq** and **Attention-based** variants, and is fully integrated with **Weights & Biases** for experiment tracking and hyperparameter tuning.

---

## Project Structure

```
da6401_assignment3-main/
├── train.py                     # Vanilla Seq2Seq training
├── train_attention.py          # Attention-based Seq2Seq training
├── model.py                    # Encoder, Decoder, Seq2Seq classes
├── dataset.py                  # Dataset class and collation function
├── predict.py                  # Run inference on trained models
├── evaluate_attention.py       # Evaluate attention-based models
├── vizualize_attention.py      # Generate attention heatmaps
├── test.ipynb                  # Jupyter notebook for testing
├── sweep.yaml                  # W&B sweep config (vanilla)
├── sweep_attention.yaml        # W&B sweep config (attention)
├── best_vanilla_model.pt       # Checkpoint for best vanilla model
├── best_attention_model.pt     # Checkpoint for best attention model
├── predictions_vanilla/        # Output predictions (vanilla)
├── predictions_attention/      # Output predictions (attention)
├── heatmaps/                   # Visual attention heatmaps
├── test_results.txt            # Output log for test cases
```

---

## Dataset

The project uses the **Tamil subset** of the [Dakshina dataset](https://huggingface.co/datasets/dakshina).

* Format: TSV with `target<TAB>input<TAB>freq`
* Example: `நான்	nAn	1`
* Location (relative):

  * `dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv`

---

## Model Architectures

### 1. Vanilla Seq2Seq

* Encoder and Decoder: LSTM / GRU
* Embedding layer per module
* No attention

### 2. Attention Seq2Seq

* Bahdanau-style attention (additive)
* Attention weights visualized per input

---
# Using the Model
## Training

```bash
# Train Vanilla Model
python train.py

# Train Attention Model
python train_attention.py
```

Model and training parameters are set via `wandb.config` and optionally tuned with `sweep.yaml`.

---

## Hyperparameter Sweeps (W\&B)

```bash
# Vanilla sweep
wandb sweep sweep.yaml
wandb agent <sweep_id>

# Attention sweep
wandb sweep sweep_attention.yaml
wandb agent <sweep_id>
```

---

## Evaluation & Prediction

```bash
# Evaluate Attention Model
python evaluate_attention.py

# Predict & Save outputs
python predict.py
```

---

## Visualizations

```bash
# Generate attention heatmaps
python vizualize_attention.py
```

Artifacts:

* `confusion_matrix_true.png`
* `attention_heatmap_grid_correct.png`

---

## Results

* **Best Val Accuracy (Vanilla):** \~0.5587
* **Best Val Accuracy (Attention):** \~0.6039
* Confusion matrix & qualitative results in `/heatmaps/` and `predictions_*/`

---

## Requirements

```bash
pip install torch pandas tqdm wandb
```

---

## License & Credits

* Dataset: [Dakshina](https://huggingface.co/datasets/dakshina)
* Course: DA6401 - Deep Learning (IIT Madras)
* Author: \[Sakthe Balan GK]

---

## Notes

* Codebase designed for easy experimentation.
* Attention visualization is useful for interpretability.
* Weights are saved automatically on best validation accuracy.

---

Enjoy exploring transliteration with Seq2Seq models!
