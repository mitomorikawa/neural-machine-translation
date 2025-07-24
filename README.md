# Neural Machine Translation (NMT) - English to French

## Overview
This is a PyTorch implementation of Neural Machine Translation for translating English text to French. I used a seq2seq RNN model with Bahdanau attention, which yielded a decent Bleu score and translations. I also wanted to implement a Transformer translation model from scratch as well (the painstaking struggle of which is visible in the codes here and there), but whichever part I tweaked the translation quality never improved. After a tormenting month of struggle I realised that there are so many nuts and bolts I need to correctly implement in order to efficiently train a Transformer model without collapsing it, and that trying to implement them all is kind of beyond my capabili
ty right now, considering my time constraint and current knowledge. So I decided to finish this after I've gained deeper understanding of transformer and these techniques, hopefully within a year before coming back to Japan.

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

