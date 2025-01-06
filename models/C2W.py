import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
from .transformers.transformer import DecoderLayer, EncoderLayer 

class CharacterToWord(nn.Module):
    def __init__(self, d_model, num_heads, num_decoder_layers, d_ff, dropout):
        super(CharacterToWord, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.char_head = nn.Linear(d_model, 102)  # 36 characters + 1 for 'not a character'
        self.ctc_decoder = nn.CTCLoss(blank=0)  # CTC loss for final word recognition

    def forward(self, char_embeddings, char_positions):
        # Ensure char_embeddings and char_positions have the correct dimensions
        batch_size, seq_length, d_model = char_embeddings.size()
        char_positions = char_positions.view(batch_size, seq_length, -1)

        for dec_layer in self.decoder_layers:
            char_embeddings = dec_layer(char_embeddings, char_positions, None, None)
        char_logits = self.char_head(char_embeddings)
        return char_logits

    def compute_loss(self, char_logits, targets, target_lengths):
        input_lengths = torch.full(size=(char_logits.size(1),), fill_value=char_logits.size(0), dtype=torch.long)
        loss = self.ctc_decoder(char_logits.log_softmax(2), targets, input_lengths, target_lengths)
        return loss
