import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = nn.Linear(configs.enc_in+1, configs.d_model)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # Encoder
        self.encoder = nn.LSTM(configs.d_model, configs.d_model, configs.e_layers, bidirectional=True, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(configs.d_model * 2, configs.d_model, configs.d_layers, bidirectional=True,
                               batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(configs.d_model * 2, configs.d_model, bias=True), nn.ReLU(),
                                          nn.Linear(configs.d_model, configs.c_out))
        self.linear = nn.Linear(configs.d_model * configs.seq_len * 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(torch.cat([x_enc, x_mark_enc], dim=-1))
        enc_out, (h, c) = self.encoder(enc_out)  # [B, L, D*2]
        EncCtextOut = enc_out[:, -1, :].unsqueeze(1)
        EncCtextOut = torch.repeat_interleave(EncCtextOut, repeats=self.pred_len, dim=1)
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, (_, _) = self.decoder(EncCtextOut)
        dec_out = self.output_layer(dec_out)

        cycle_distance_out = self.linear(enc_out.reshape(enc_out.shape[0], -1))
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], 0, 0, cycle_distance_out
        else:
            return dec_out[:, -self.pred_len:, :], 0, cycle_distance_out  # [B, L, D]
