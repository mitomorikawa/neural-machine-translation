import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import library.translator as translator
import library.nn_architectures as nn_architectures
import library.preprocessor as preprocessor
import torch

encoder_path = "../models/bahdanau_encoder_20250709_125813_2"
decoder_path = "../models/bahdanau_decoder_20250709_125814_2"
src_language = "English"

translator_class = translator.Translator(encoder_path, decoder_path)
encoder = nn_architectures.RNNEncoder(9783, 1024)
decoder = nn_architectures.RNNDecoder(1024, 15532, 69)

encoder, decoder = translator_class.load_models(encoder, decoder)
src_vocab = translator_class.load_vocab("../data/eng_vocab.pkl")
tgt_vocab = translator_class.load_vocab("../data/fra_vocab.pkl")

src_word2idx = src_vocab["word2idx"]
tgt_idx2word = tgt_vocab["idx2word"]

src_text = input(f"Enter a text in {src_language}: ")

standadizer = preprocessor.Standardizer()
src_standadized_text = standadizer.standardize([src_text])

tokenizer = preprocessor.Tokenizer()    
src_tokenized_text = tokenizer.word_tokenize(src_standadized_text, 55)

indexer = preprocessor.Indexer()
indexer.word2idx = src_word2idx
src_idx = torch.tensor(indexer.text_to_indices(src_tokenized_text, verbose=False))

translated_text = translator_class.translate(src_idx, encoder, decoder, tgt_idx2word)
print(f"Translated text: {translated_text}")
