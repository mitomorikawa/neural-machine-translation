import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import library.trainer as trainer
import library.translator as translator

load = trainer.TensorLoader()
eng = load.load("../data/eng_train.pt", device="cpu")
fra = load.load("../data/fra_train.pt", device="cpu")
translate = translator.Translator(
    encoder_path="../models/transformer_encoder_20250706_094437_1",
    decoder_path="../models/transformer_decoder_20250706_094437_1"
)
eng_vocab = translate.load_vocab("../data/eng_vocab.pkl")
fra_vocab = translate.load_vocab("../data/fra_vocab.pkl")

for i in range(20):
    print(f"English: {eng[i]}")
    print(f"sentence: {' '.join([eng_vocab['idx2word'][word.item()] for word in eng[i]])}")
    print(f"French: {fra[i]}")
    print(f"sentence: {' '.join([fra_vocab['idx2word'][word.item()] for word in fra[i]])}")
    print("-" * 50)