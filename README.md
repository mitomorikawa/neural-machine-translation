# Neural Machine Translation (NMT) - English to French

A PyTorch implementation of Neural Machine Translation using the Bahdanau attention mechanism for translating English text to French.

## Features

- Encoder-Decoder architecture with Bahdanau attention
- Batch processing support
- TensorBoard integration for training visualization
- Modular design with separate preprocessing, training, validation, and translation modules

## Dataset

The dataset was taken from https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench.

## Project Structure

```
nmt/
├── data/                  # Preprocessed data and vocabularies
│   ├── eng_fra.csv       # Original English-French parallel corpus
│   ├── eng_*.pt          # Preprocessed English tensors
│   ├── fra_*.pt          # Preprocessed French tensors
│   └── *_vocab.pkl       # Vocabulary mappings
├── models/               # Saved model checkpoints
├── runs/                 # TensorBoard logs
├── src/                  # Source code
│   ├── bahdanau_train.py # Main training script
│   ├── preprocess.py     # Data preprocessing script
│   ├── translate.py      # Interactive translation script
│   ├── validate.py       # Model validation script
│   └── library/          # Core modules
│       ├── nn_architectures.py  # Neural network models
│       ├── preprocessor.py      # Data preprocessing utilities
│       ├── trainer.py           # Training logic
│       ├── translator.py        # Translation utilities
│       └── validator.py         # Validation utilities
└── tests/                # Unit tests

```

## Requirements

- Python 3.x
- PyTorch
- TensorBoard
- NumPy
- Pickle

## Usage

### 1. Data Preprocessing

Preprocess the raw English-French parallel corpus:

```bash
cd src
python preprocess.py
```

This will:
- Load the CSV file containing English-French sentence pairs
- Standardize and tokenize the text
- Build vocabularies for both languages
- Split data into train/validation/test sets
- Save preprocessed tensors and vocabulary files

### 2. Training

Train the NMT model with Bahdanau attention:

```bash
python bahdanau_train.py
```

Default configuration:
- Hidden size: 128
- Learning rate: 0.001
- Batch size: 16
- Epochs: 100
- Optimizer: Adam
- Loss function: CrossEntropyLoss

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=../runs/training_logs
```

### 3. Translation

Translate English text to French interactively:

```bash
python translate.py
```

Enter English text when prompted, and the model will output the French translation.

This will calculate metrics like BLEU score and perplexity on the validation/test data.

## Model Architecture

### Encoder
- Bidirectional GRU-based RNN
- Input embedding layer
- Hidden state initialization

### Decoder with Bahdanau Attention
- GRU-based RNN with attention mechanism
- Attention weights computed using encoder hidden states
- Context vector combined with decoder input
- Output projection layer

## Testing

Run unit tests:

```bash
cd tests
python -m pytest
```

## Model Checkpoints

Trained models are saved in the `models/` directory with timestamps:
- `bahdanau_encoder_YYYYMMDD_HHMMSS_epoch`
- `bahdanau_decoder_YYYYMMDD_HHMMSS_epoch`

## Notes

- The model uses teacher forcing during training
- Vocabulary includes special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- GPU acceleration is automatically used if available

