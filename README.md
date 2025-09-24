# Neural Machine Translation (NMT) - English to French

## Overview
This is a PyTorch implementation of Neural Machine Translation for translating English text to French. I implemented a seq2seq Bahdanau attention RNN model and a seq2seq transformer model (the latter still needs some debugging)

## Dataset
The dataset was taken from https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench. 

## Models
### 1. Bahdanau Attention

Dataset used: https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench. 
Hyperparameters
•hidden_size = 1024  
•batch_size = 256
•learning rate = 0.001
•Encoder used bidirectional GRU

### 2. Transformer



## Project Structure

```
nmt/
├── data/                         # Preprocessed data and vocabularies
├── models/                       # Saved model checkpoints
├── runs/                         # TensorBoard logs
├── src/                          # Source code
│   ├── bahdanau_train.py         # Bahdanau attention model training
│   ├── bahdanau_translate.py     # Bahdanau model translation
│   ├── transformer_train.py      # Transformer model training
│   ├── transformer_translate.py  # Transformer model translation
│   ├── preprocess.py             # Data preprocessing script
│   ├── validate.py               # Model validation script
│   └── library/                  # Core modules
│       ├── nn_architectures.py   # Neural network models
│       ├── preprocessor.py       # Data preprocessing utilities
│       ├── trainer.py            # Training logic
│       ├── translator.py         # Translation utilities
│       └── validator.py          # Validation utilities
└── tests/                        # Unit tests
    ├── test_large.py             # Tests for large dataset
    └── library/                  # Library module tests
        ├── test_nn_architectures.py
        ├── test_preprocessor.py
        ├── test_trainer.py
        └── test_translator.py

```

## Requirements

- Python 3.x
- PyTorch
- TensorBoard
- NumPy
- Pickle

## Training

### 1. Data Preprocessing

Place the csv file containing sentence pairs in data/ and run

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
```
python bahdanau_train.py
```
or 
```
python transformer_train.py (This one needs fixing)
```

### 3. Translation
run
```
python bahdanau_translate.py
```
or 
```
python transformer_translate.py (Needs debugging)
```

### Current Results:
RNN: Read https://github.com/peuape/neural-machine-translation/blob/main/runs/validation/bahdanau_val.txt
Transformer: Not finished yet.

