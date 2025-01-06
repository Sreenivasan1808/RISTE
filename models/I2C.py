from .resnet import ResNetBackbone
from .transformers.transformer import *
from torch import nn
import torch
# from C2W import CharacterToWord
# from utils import positionalencoding2d, PositionalEncoding2D

# Complete I2C Model with ResNet-50 Backbone and Transformer Encoder-Decoder
class I2CModel(nn.Module):
    def __init__(self, num_heads=8, embed_dim=512, ff_dim=1024, num_encoder_layers=3, num_decoder_layers=1, num_classes=37):
        super(I2CModel, self).__init__()
        self.cnn_backbone = ResNetBackbone()  # ResNet-50 Backbone
        
        
        # Positional Encoding (can be customized)
        self.positional_encoding = PositionalEncoding2D(embed_dim,)
        # self.positional_encoding = positionalencoding2d()
        
        # Transformer Encoder stack (I2C)
        # self.encoder_layers = nn.ModuleList([
        #     TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(num_encoder_layers)
        # ])
        
        # self.encoder_layers = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8), num_layers=3
        # )
        
        # Transformer Decoder stack
        # self.decoder_layers = nn.ModuleList([
        #     TransformerDecoder(embed_dim, num_heads, ff_dim) for _ in range(num_decoder_layers)
        # ])

        self.transformer = Transformer()

        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8), num_layers=1
        )
        
        # Output layers for character class and position
        self.char_class_output = nn.Linear(embed_dim, num_classes)
        self.char_position_output = nn.Linear(embed_dim, 2)  # x, y coordinates

        # Xavier initialization for I2C and C2W weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, img, tgt):
        # CNN Backbone to extract features from image (ResNet-50)
        img_features = self.cnn_backbone(img)

        print("Feature map size: ", img_features.shape)
        
        # Add positional encoding
        img_features = img_features + self.positional_encoding
        
        # Transformer Encoder (I2C)
        for encoder in self.encoder_layers:
            img_features = encoder(img_features.unsqueeze(0)).squeeze(0)
        
        # Transformer Decoder 
        decoder_output = tgt
        for decoder in self.decoder_layers:
            decoder_output = decoder(decoder_output.unsqueeze(0), img_features.unsqueeze(0)).squeeze(0)
        
        # Output heads for character class and position
        char_class_pred = self.char_class_output(decoder_output)
        char_position_pred = self.char_position_output(decoder_output)
        
        return char_class_pred, char_position_pred
 
class ImageToCharacter(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, height, width, max_length, dropout):
        super(ImageToCharacter, self).__init__()
        self.cnn_backbone = ResNetBackbone(d_model)
        self.positional_encoding = PositionalEncoding2D(d_model, height, width)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.char_head = nn.Linear(d_model, 102)  # 36 characters + 1 for 'not a character'
        self.pos_head = nn.Linear(d_model, 200)  # 25 positions + 1 for 'not belongs to word'
        self.dropout = nn.Dropout(dropout)

        # Learn queries dynamically based on max_length
        self.char_queries = nn.Parameter(torch.randn(1, max_length, d_model))  # 1 for broadcast

        # Add C2W module 
        # self.c2w = CharacterToWord(d_model, num_heads, num_decoder_layers, d_ff, dropout)

    def forward(self, x):
        x = self.cnn_backbone(x)  # Output: [batch_size, d_model, H, W]
        # print("\n\nFeature maps size: ", x.shape)
        # print("Feature maps: ", x)
        x = self.positional_encoding(x)  # Add positional encoding
        # print("\n\nFeatures added with positional encoding size: ", x.shape)

        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1).transpose(1, 2)  # Flatten H, W

        for enc_layer in self.encoder_layers:
            x = enc_layer(x, None)

        char_queries = self.char_queries.expand(batch_size, -1, -1)  # Expand for batch size
        for dec_layer in self.decoder_layers:
            char_queries = dec_layer(char_queries, x, None, None)

        # print("Positional embeddings size: ", char_queries.shape)
        char_logits = self.char_head(char_queries)  # [batch_size, max_length, 37]
        pos_logits = self.pos_head(char_queries)    # [batch_size, max_length, 26]
        # word_logits = self.c2w(char_queries, pos_logits)

        return char_logits, pos_logits



