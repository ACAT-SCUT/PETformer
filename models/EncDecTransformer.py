import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
        Long Time Series Forecasting via Encoder-Decoder Transformer
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

        # computer token sizes
        self.input_token_size = int((self.seq_len - self.win_len) / self.win_stride + 1)
        self.output_token_size = int(self.pred_len / self.win_len) + 1

        # build model
        self.mapping = nn.Linear(self.win_len, self.d_model)
        self.positionEmbbeding = nn.Embedding(self.input_token_size, self.d_model)
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True,
                                                     dim_feedforward=self.d_model * self.ff_factor,
                                                     dropout=self.dropout_rate), num_layers=2)
        self.transformerDecoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True,
                                                     dim_feedforward=self.d_model * self.ff_factor,
                                                     dropout=self.dropout_rate), num_layers=2)
        self.outputPositionEmbbeding = nn.Embedding(self.output_token_size, self.d_model)

        self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1, batch_first=True, dropout=0.5)
        self.channelBatchNorm = nn.BatchNorm1d(self.enc_in)
        self.predict = nn.Linear(self.d_model, self.win_len)

        self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [Batch, Input length, Channel]
        # norm
        x = self.revinLayer(x_enc, 'norm').permute(0, 2, 1)

        # mapping
        inputTokens = self.mapping(x.unfold(dimension=-1, size=self.win_len, step=self.win_stride))  # b c it d

        # add position
        batch_index = torch.arange(self.input_token_size).expand(x_enc.size(0) * self.enc_in, self.input_token_size).to(
            x.device)
        inputTokens = inputTokens.view(-1, self.input_token_size, self.d_model) + self.positionEmbbeding(
            batch_index)  # b*c,t,d

        # transformer extrac feature
        inputTokens = self.transformerEncoder(inputTokens)

        # build output tokens
        outputTokens = self.mapping(self.revinLayer(x_dec, 'norm').permute(0, 2, 1).unfold(dimension=-1, size=self.win_len, step=self.win_stride))
        # outputTokens = torch.cat(
        #     [labelToken, torch.zeros(labelToken.size(0), self.output_token_size - 1, self.d_model)], dim=1)
        output_batch_index = torch.arange(self.output_token_size).expand(x_enc.size(0) * self.enc_in,
                                                                         self.output_token_size).to(x.device)
        outputTokens = outputTokens.view(-1, self.output_token_size, self.d_model) + self.outputPositionEmbbeding(output_batch_index)

        # transformer decoder
        outputTokens = self.transformerDecoder(outputTokens, inputTokens)

        # get output
        outputTokens = outputTokens[:, 1:, :].reshape(-1, self.enc_in, self.output_token_size-1, self.d_model)

        # mix channel if 1
        if self.channel_attn == 1:
            outputTokens = outputTokens.permute(0, 2, 1, 3).reshape(-1, self.enc_in, self.d_model)
            outputTokens = self.channelBatchNorm(self.attn(outputTokens, outputTokens, outputTokens)[0]) + outputTokens
            outputTokens = outputTokens.view(-1, self.output_token_size-1, self.enc_in, self.d_model).permute(0, 2, 1, 3)

        y = self.predict(outputTokens).view(-1, self.enc_in, self.pred_len).permute(0, 2, 1)  # b,s,c

        # denorm
        y = self.revinLayer(y, 'denorm')

        return y
