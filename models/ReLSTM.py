import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    LSTM
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.lstm = nn.LSTM(input_size=1, hidden_size=30, num_layers=1, bidirectional=False)  # LSTM层
        self.lstmfc = nn.Linear(in_features=30, out_features=self.pred_len)

        self.fc = nn.Conv1d(96, self.pred_len, kernel_size=1, stride=1, bias=False)

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
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        x_enc = batch_x [0-96] [32,96,1]
        x_mark_enc = batch_x_mark [0-96]时间
        x_dec = dec_inp [48-96-144] 前部分label，后部分0
        x_mark_dec = batch_y_mark  [48-144]时间
        enc_self_mask = batch_y [48-144]
        """

        """
        LSTM
        """
        lstm_out, _ = self.lstm(x_enc)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.lstmfc(lstm_out).unsqueeze(dim=-1)

        # x = self.fc(x_enc)
        return lstm_out

        return self.fc(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [32*96*512]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            dec_out = self.fc(dec_out)
            return dec_out
            # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
