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

        self.lstm = nn.LSTM(input_size=1, hidden_size=13, num_layers=1, bidirectional=False)  # LSTM层
        self.lstmfc = nn.Linear(in_features=13, out_features=self.pred_len)

        self.distribution_mu = nn.Linear(13, self.pred_len)
        self.distribution_presigma = nn.Linear(13, self.pred_len)
        self.distribution_sigma = nn.Softplus()

        self.fc = nn.Conv1d(96, self.pred_len, kernel_size=1, stride=1, bias=False)

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

        pre_sigma = self.distribution_presigma(lstm_out)
        mu = self.distribution_mu(lstm_out)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        return mu.unsqueeze(dim=-1)

        # lstm_out = self.lstmfc(lstm_out).unsqueeze(dim=-1)

        # return lstm_out

