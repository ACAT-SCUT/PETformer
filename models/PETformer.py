import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
        Long Time Series Forecasting via Placeholder-enhanced Transformer
            · A shared placeholder occupies the output window to be predicted and is entered into the Transformer encoder for feature learning
            · A Long Sub-sequence Division strategy is utilized to improve the semantic richness of the Transformer tokens
            · Several information-interaction modes based on channel independent strategy is explored
            · A token-wise prediction layer is utilized to greatly reduce the number of learnable parameters
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.ff_factor = configs.ff_factor
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.dropout_rate = configs.dropout
        self.win_len = configs.win_len
        self.win_stride = configs.win_stride
        self.channel_attn = configs.channel_attn
        self.attn_type = configs.attn_type

        # compute token sizes
        self.input_token_size = int((self.seq_len - self.win_len) / self.win_stride + 1)
        self.output_token_size = int(self.pred_len / self.win_len)
        self.all_token_size = self.input_token_size + self.output_token_size

        # build model
        self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=True)
        self.mapping = nn.Linear(self.win_len, self.d_model)
        self.placeholder = nn.Parameter(torch.randn(self.d_model))
        self.positionEmbbeding = nn.Embedding(self.all_token_size, self.d_model)
        self.transformerEncoderLayers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True,
                                       dim_feedforward=self.d_model * self.ff_factor,
                                       dropout=self.dropout_rate) for i in range(self.e_layers))
        self.batchNorm = nn.ModuleList(nn.BatchNorm1d(self.all_token_size) for i in range(self.e_layers))
        self.predict = nn.Linear(self.d_model, self.win_len)

        # masking method of attention: 0:all2all; 1:no inter-futuer; 2:no inter-history; 3:neither inter- future nor history
        if self.attn_type == 0:
            self.mask = torch.full((self.all_token_size, self.all_token_size), bool(False))
        elif self.attn_type == 1:
            self.mask = torch.full((self.all_token_size, self.all_token_size), bool(True))
            self.mask[:, :self.input_token_size] = False
            self.mask[torch.eye(self.all_token_size, dtype=bool)] = False
        elif self.attn_type == 2:
            self.mask = torch.full((self.all_token_size, self.all_token_size), bool(True))
            self.mask[self.input_token_size:, :] = False
            self.mask[torch.eye(self.all_token_size, dtype=bool)] = False
        elif self.attn_type == 3:
            self.mask = torch.full((self.all_token_size, self.all_token_size), bool(True))
            self.mask[self.input_token_size:, :self.input_token_size] = False
            self.mask[torch.eye(self.all_token_size, dtype=bool)] = False
        else:
            self.mask = torch.full((self.all_token_size, self.all_token_size), bool(False))

        # information interaction mode between different channels:  0: no channel attn; 1:self-attn; 2:cross-attn; 3:channel identity
        if self.channel_attn != 0:
            self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1, batch_first=True, dropout=0.5)
            self.channelBatchNorm = nn.BatchNorm1d(self.enc_in)

        if self.channel_attn == 2 or self.channel_attn == 3:
            self.channel_pe = nn.Embedding(self.enc_in, self.d_model)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [Batch, Input length, Channel]
        # norm
        x = self.revinLayer(x_enc, 'norm').permute(0, 2, 1)


        # mapping
        inputTokens = self.mapping(x.unfold(dimension=-1, size=self.win_len, step=self.win_stride))

        # build tokens
        allTokens = torch.cat(
            (inputTokens, self.placeholder.repeat(x_enc.size(0), self.enc_in, self.output_token_size, 1)), dim=2)

        # add position
        batch_index = torch.arange(self.all_token_size).expand(x_enc.size(0) * self.enc_in, self.all_token_size).to(
            x.device)
        allTokens = allTokens.view(-1, self.all_token_size, self.d_model) + self.positionEmbbeding(batch_index)

        # transformer encoder extrac feature
        for i in range(self.e_layers):
            allTokens = self.batchNorm[i](
                self.transformerEncoderLayers[i](allTokens, src_mask=self.mask.to(x.device))) + allTokens

        # get outputTokens
        outputTokens = allTokens.view(-1, self.enc_in, self.all_token_size, self.d_model)[:, :, self.input_token_size:, :]

        # information interaction between different channels: 0: no channel attn; 1:self-attn; 2:cross-attn; 3:channel identity
        if self.channel_attn == 1:
            outputTokens = outputTokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            outputTokens = self.channelBatchNorm(self.attn(outputTokens, outputTokens, outputTokens)[0]) + outputTokens
            outputTokens = outputTokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)
        elif self.channel_attn == 2:
            channel_index = torch.arange(self.enc_in).expand(x_enc.size(0) * self.output_token_size, self.enc_in).to(
                x.device)
            Query = self.channel_pe(channel_index)
            outputTokens = outputTokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            outputTokens = self.channelBatchNorm(self.attn(Query, Query, outputTokens)[0]) + outputTokens
            outputTokens = outputTokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)
        elif self.channel_attn == 3:
            channel_index = torch.arange(self.enc_in).expand(x_enc.size(0) * self.output_token_size, self.enc_in).to(
                x.device)
            channelIdentity = self.channel_pe(channel_index)
            outputTokens = outputTokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            outputTokens = outputTokens + channelIdentity
            outputTokens = outputTokens.view(-1, self.output_token_size, self.enc_in, self.d_model).permute(0, 2, 1, 3)

        # predict
        y = self.predict(outputTokens).view(-1, self.enc_in, self.pred_len).permute(0, 2, 1)  # b,s,c

        # denorm
        y = self.revinLayer(y, 'denorm')

        return y
