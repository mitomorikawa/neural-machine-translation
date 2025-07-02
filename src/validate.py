import library.validator as validator
import library.translator as translator
import library.trainer as trainer
import library.nn_architectures as nn_architectures
import torch

encoder_path = "../models/bahdanau_encoder_20250628_122604_2"
decoder_path = "../models/bahdanau_decoder_20250628_122604_2"


hidden_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = nn_architectures.RNNEncoder(9783, hidden_size).to(device)
decoder = nn_architectures.RNNDecoder(hidden_size, 15532, 69).to(device)
translator_instance = translator.Translator(encoder_path, decoder_path)
encoder, decoder = translator_instance.load_models(encoder, decoder)

loader = trainer.TensorLoader()
src_val = loader.load("../data/eng_val.pt")
tgt_val = loader.load("../data/fra_val.pt")


src_vocab_path = "../data/eng_vocab.pkl"
tgt_vocab_path = "../data/fra_vocab.pkl"


src_vocab = translator_instance.load_vocab(src_vocab_path)
tgt_vocab = translator_instance.load_vocab(tgt_vocab_path)
src_idx2word = src_vocab["idx2word"]
tgt_idx2word = tgt_vocab["idx2word"]

validator_instance = validator.Validator(
    encoder=encoder,
    decoder=decoder,
    translator=translator_instance,
    src_val=src_val,
    tgt_val=tgt_val,
    src_idx2word=src_idx2word,
    tgt_idx2word=tgt_idx2word
)
validator_instance.evaluate_val_set()
translate_random_samples = validator_instance.translate_random_samples()