""" 
This module provides a validation class.


"""

from bleu import list_bleu
from tqdm import tqdm
class Validator:
    def __init__(self, encoder,decoder, translator, src_val, tgt_val, src_idx2word, tgt_idx2word):
        self.encoder = encoder  
        self.decoder = decoder
        self.translator = translator
        self.src_val = src_val
        self.tgt_val = tgt_val
        self.src_idx2word = src_idx2word
        self.tgt_idx2word = tgt_idx2word

    def evaluate_val_set(self):
        """ 
        Evaluates the validation set using BLEU score.
        params:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            translator (Translator): The translator instance.
            src_val (torch.Tensor): The source validation data.
            tgt_val (torch.Tensor): The target validation data.

        """
  
        total_bleu = 0.0
        for src, tgt in tqdm(zip(self.src_val, self.tgt_val)):
            src = src.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
            translated = self.translator.translate(src, self.encoder, self.decoder, self.tgt_idx2word)
            reference_words = [self.tgt_idx2word[idx] for idx in tgt[0].tolist() if idx not in [0, 1, 2]]  # Skip PAD, EOS, BOS tokens
            reference = ' '.join(reference_words)

            bleu_score = list_bleu([[reference]], [translated])
            print(f"Source: {src}, Translated: {translated}, Reference: {reference}, BLEU Score: {bleu_score:.4f}")
            total_bleu += bleu_score
        avg_bleu = total_bleu / len(self.src_val)


        print(f"Average BLEU score on validation set: {avg_bleu:.4f}")

    def translate_random_samples(self, num_samples=10):
        """ 
        Translates random samples from the validation set.
        params:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            translator (Translator): The translator instance.
            src_val (torch.Tensor): The source validation data.
            tgt_idx2word (dict): The target index to word mapping.
            num_samples (int): Number of random samples to translate.

        """
        import random
        indices = random.sample(range(len(self.src_val)), num_samples)
        for idx in indices:
            src = self.src_val[idx].unsqueeze(0)
            src_sentence = [self.src_idx2word[i.item()] for i in self.src_val[idx] if i not in [0, 1, 2]]
            src_sentence = ' '.join(src_sentence)
            translated = self.translator.translate(src, self.encoder, self.decoder, self.tgt_idx2word)
            print(f"Source: {src_sentence}, Translated: {translated}")
