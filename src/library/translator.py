
import torch
import pickle

class Translator:
    """
    This class is responsible for translating texts using a trained_model

    attributes:
        - encoder_path (str): Path to the encoder model.
        - decoder_path (str): Path to the decoder model.
    """
    def __init__(self, encoder_path: str, decoder_path: str, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.device = device

    def load_models(self, encoder, decoder):
        """ 
        Loads the encoder and decoder models from the specified paths.
        Args:
            encoder (nn.Module): An initialised encoder model.
            decoder (nn.Module): An initialised decoder model.

        returns:
            encoder (nn.Module): The loaded encoder model.
            decoder (nn.Module): The loaded decoder model.
        """
        encoder.load_state_dict(torch.load(self.encoder_path, map_location=self.device,weights_only=True))
        decoder.load_state_dict(torch.load(self.decoder_path, map_location=self.device,weights_only=True))
        encoder.eval()
        decoder.eval()
        return encoder, decoder
    
    def load_vocab(self, vocab_path):
        """ 
        Loads the vocabulary from the specified path.
        
        Args:
            vocab_path (str): Path to the vocabulary file.

        returns:
            dict: The loaded vocabulary.
        """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return vocab
    
    def translate(self, src_idx, encoder, decoder, tgt_idx2word, eos_token=1, transformer=False):    
        """ 
        Translates the source text using the encoder and decoder models.
        
        Args:
            src_idx (List[List[int]]): The source text indices to translate.
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model. 

        returns:
            List[str]: The translated text.
        """
        # This method should implement the translation logic
        # For now, it is a placeholder

        with torch.no_grad():
            encoder_outputs, encoder_hidden = encoder(src_idx)
            if transformer:
                decoder_outputs = decoder.infer_greedy(src_idx, encoder_outputs)
                translated_indices = decoder_outputs.squeeze(0).tolist()
                decoded_words = []
                        # For transformer, translated_indices is a 1D list of token IDs
                for idx in translated_indices:
                    if idx == 0 or idx == 2:
                        continue
                    elif idx == eos_token:
                        break
                    decoded_words.append(tgt_idx2word[idx])
            else:
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, greedy=False, beam_width=5)
                _, topi = decoder_outputs.topk(1)
                translated_indices = topi.squeeze(1).tolist()
                decoded_words = []
            
                for idx in translated_indices[0]:
                    if idx[0] == 0 or idx[0] == 2:
                        continue
                    elif idx[0]== eos_token:
                        break
                    decoded_words.append(tgt_idx2word[idx[0]])
            translated_text = ' '.join(decoded_words)
            return translated_text         
            
        
        