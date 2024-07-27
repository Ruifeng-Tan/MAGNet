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
        self.enc_embedding = nn.Linear(3, configs.d_model)

        self.lstm = nn.LSTM(configs.d_model, configs.d_model, 2, bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(configs.d_model * 2, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(torch.cat([x_dec, x_mark_dec], dim=-1))
        enc_out, (h, c) = self.lstm(enc_out)
        dec_out = self.output_layer(enc_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], 0, 0, 0
        else:
            return dec_out[:, -self.pred_len:, :], 0, 0  # [B, L, D]
