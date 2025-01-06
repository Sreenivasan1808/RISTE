import torch
import torch.nn as nn
from .resnet import ResNetBackbone
from .I2C import ImageToCharacter
from .C2W import CharacterToWord

class I2C2W(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, height, width, max_length, dropout, num_classes=37):
        super(I2C2W, self).__init__()
        self.i2c = ImageToCharacter(d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, height, width, max_length, dropout)
        self.c2w = CharacterToWord(d_model, num_heads, num_decoder_layers, d_ff, dropout)

    def forward(self, x):
        char_logits, pos_logits = self.i2c(x)
        word_logits = self.c2w(char_logits, pos_logits)
        return char_logits, pos_logits, word_logits

    def compute_loss(self, char_logits, pos_logits, word_logits, targets, target_lengths):
        char_class_loss = nn.CrossEntropyLoss(ignore_index=0)(char_logits.view(-1, char_logits.size(-1)), targets['char_classes'].view(-1))
        char_position_loss = nn.CrossEntropyLoss(ignore_index=-1)(pos_logits.view(-1, pos_logits.size(-1)), targets['char_positions'].view(-1))
        word_loss = self.c2w.compute_loss(word_logits, targets['char_classes'], target_lengths)
        total_loss = char_class_loss + char_position_loss + word_loss
        return total_loss
