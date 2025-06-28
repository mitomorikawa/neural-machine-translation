import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import src.library.translator as translator
import src.library.nn_architectures as nn_architectures
import src.library.trainer as trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_loadmodels():
    test_translator = translator.Translator("../../models/bahdanau_encoder_20250625_185000_22", "../../models/bahdanau_decoder_20250625_185000_22")
    encoder = nn_architectures.EncoderRNN(9783, 128)
    decoder = nn_architectures.AttnDecoderRNN(128, 15532, 69)
 
    encoder, decoder = test_translator.load_models(encoder, decoder)    
    if not isinstance(encoder, nn_architectures.EncoderRNN) or not isinstance(decoder, nn_architectures.AttnDecoderRNN):
        raise AssertionError("Encoder model is not of type EncoderRNN")
    print("Model loader test passed.")

def test_load_vocab():
    test_translator = translator.Translator("../../models/bahdanau_encoder_20250625_185000_22", "../../models/bahdanau_decoder_20250625_185000_22")
    vocab = test_translator.load_vocab("../../data/eng_vocab.pkl")
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    
    if (word2idx["<sos>"] == 0 and 
        word2idx["<eos>"] == 1 and 
        word2idx["<pad>"] == 2 and 
        word2idx["<unk>"] == 3 and 
        word2idx["hi"] == 4 and
        word2idx["."] == 5 and
        idx2word[0] == "<sos>" and 
        idx2word[1] == "<eos>" and 
        idx2word[2] == "<pad>" and 
        idx2word[3] == "<unk>" and
        idx2word[4] == "hi" and
        idx2word[5] == "."
        ):
        print("Vocabulary loader test passed.")
    else:
        print(f"""word2idx[\"<sos>\"]: {word2idx['<sos>']}, 
                  word2idx[\"<eos>\"]: {word2idx['<eos>']},
                    word2idx[\"<pad>\"]: {word2idx['<pad>']},
                    word2idx[\"<unk>\"]: {word2idx['<unk>']},
                    word2idx[\"hi\"]: {word2idx['hello']},
                    word2idx[\".\"]: {word2idx['.']},
                    idx2word[0]: {idx2word[0]},
                    idx2word[1]: {idx2word[1]},
                    idx2word[2]: {idx2word[2]},
                    idx2word[3]: {idx2word[3]},
                    idx2word[4]: {idx2word[4]},
                    idx2word[5]: {idx2word[5]}
              """)
        raise AssertionError("Vocabulary does not contain expected tokens or indices.")
    


def test_translate():
    test_translator = translator.Translator("../../models/bahdanau_encoder_20250625_185000_22", "../../models/bahdanau_decoder_20250625_185000_22")
    encoder = nn_architectures.EncoderRNN(9783, 128)
    decoder = nn_architectures.AttnDecoderRNN(128, 15532, 69)
    tensorloader = trainer.TensorLoader()
    src_idx = tensorloader.load("../../data/eng_train.pt", device)
    tgt_idx = tensorloader.load("../../data/eng_train.pt", device)
    
    src_idx = torch.unsqueeze(src_idx[10], 0)  
    #tgt_idx = torch.unsqueeze(tgt_idx[10], 0)  
    encoder, decoder = test_translator.load_models(encoder, decoder)  
    src_vocab = test_translator.load_vocab("../../data/eng_vocab.pkl")
    tgt_vocab = test_translator.load_vocab("../../data/fra_vocab.pkl")
    

    translated_text = test_translator.translate(src_idx, encoder, decoder, tgt_vocab["idx2word"])
    if type(translated_text) is not str:
        raise AssertionError("Translated text is not a string")
    print(f"Translation test passed: {' '.join([src_vocab['idx2word'][idx.item()] for idx in src_idx[0]])}->{translated_text}")

if __name__ == "__main__":
    test_loadmodels()
    test_load_vocab()
    test_translate()
    print("All tests passed.")