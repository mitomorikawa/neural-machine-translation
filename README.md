# Neural Machine Translation (NMT) - English to French

## Overview
This is a PyTorch implementation of Neural Machine Translation for translating English text to French. For training, I used Bahdanau attention and Transformer, which are implemented from scratch in src/library/nn_architectures.py. 

## Dataset

The dataset was taken from https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench. and WMT 2014 English-French.

## Models
### 1. Bahdanau Attention

Dataset used: https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench. 
Hyperparameters
•hidden_size = 1024  
•batch_size = 256
•learning rate = 0.001
•Encoder used bidirectional GRU

### 2. Transformer

Hyperparameters
•


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

## Training

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
