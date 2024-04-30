import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class PhysicalModule(nn.Module):
    def __init__(self, configs):
        super(PhysicalModule, self).__init__()
        self.upper_limit = 5
        self.lower_limit = 0
        self.linear_en = nn.Linear(configs.d_model * configs.seq_len, 4)
        self.linear_time = nn.Linear(configs.seq_len + configs.pred_len, 4)
        self.linear_final = nn.Linear(8, 4)
        # self.sigmoid = nn.Sigmoid()
        self.std = configs.std
        self.mean = configs.mean
        self.nominal_capacity = configs.nominal_capacity
        self.max_Ed_in_train = configs.max_Ed_in_train
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        # 1. 根据网络层的不同定义不同的初始化方式
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, enc_out, cost_time):
        out = self.linear_en(enc_out.reshape(enc_out.shape[0], -1))
        # out2 = self.linear_time(cost_time.reshape(cost_time.shape[0], -1))
        # out = torch.cat([out1, out2], dim=-1)
        # out = self.linear_final(out)
        # out = self.sigmoid(out) * 5
        scale_factors = torch.ones_like(out[:, :2]).unsqueeze(1)  # [B,1, 2]
        out = out.reshape(-1, 2, 2)  # [B, 2, 2]

        As = out[:, :, 0].unsqueeze(1)  # [B, 1, 2]
        Bs = out[:, :, 1].unsqueeze(1)  # [B, 1, 2]
        lower_limit = torch.ones_like(As) * self.lower_limit
        # upper_limit = torch.ones_like(As) * self.upper_limit
        Bs = torch.where(Bs < lower_limit, lower_limit, Bs)
        # Bs = torch.where(Bs > upper_limit, upper_limit, Bs)
        scale_factors[:, :, 0] = self.nominal_capacity
        scale_factors[:, :, 1] = self.max_Ed_in_train
        cost_time = cost_time.unsqueeze(-1)
        cost_time = cost_time.repeat(1, 1, 2)  # [B, L, 2]
        pred_ratio = 1 - As * (cost_time.pow(Bs))  # [B, L, 2]
        # pred_ratio = torch.where(torch.isnan(pred_ratio), torch.full_like(pred_ratio, 1), pred_ratio)

        # pred_ratio = torch.where(pred_ratio > upper_limit, upper_limit, pred_ratio)
        pred = pred_ratio * scale_factors
        means = torch.ones_like(pred)
        stds = torch.ones_like(pred)
        means[:, :, 0], stds[:, :, 0] = self.mean[0], self.std[0]
        means[:, :, 1], stds[:, :, 1] = self.mean[1], self.std[1]
        pred = (pred - means) / stds
        return pred, As, Bs


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.std = configs.std
        self.mean = configs.mean
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # PI module
        self.PIModule = PhysicalModule(configs)
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor2, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor2, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cost_time,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # decoder
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # Physics
        PI_out, As, BS = self.PIModule(enc_out, cost_time)
        out = PI_out
        if self.output_attention:
            return out[:, -self.pred_len:, :], attns, As, BS
        else:
            return out[:, -self.pred_len:, :], As, BS  # [B, L, D]
