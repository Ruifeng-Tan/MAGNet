import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla LSTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.lstm = nn.LSTM(configs.d_model, configs.d_model, 1, bidirectional=True, batch_first=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, (h, c) = self.encoder(enc_out)
        enc_out = torch.cat([enc_out.permute(1, 0, 2)[-1].unsqueeze(0), enc_out.permute(1, 0, 2)[-1].unsqueeze(0)],
                            dim=0)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, (_, _) = self.decoder(dec_out, (enc_out, enc_out))
        dec_out = self.output_layer(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], 0, 0, 0
        else:
            return dec_out[:, -self.pred_len:, :], 0, 0  # [B, L, D]
