import torch
import torch.nn as nn
from layers.RevIN import RevIN

class Model(nn.Module):
    """
        Long Time Series Forecasting via Flattening Transformer
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
        self.input_token_size = int((self.seq_len-self.win_len)/self.win_stride+1)

        # build model
        self.mapping = nn.Linear(self.win_len, self.d_model)
        self.positionEmbbeding = nn.Embedding(self.input_token_size, self.d_model)
        self.transformerEncoderLayers = nn.ModuleList(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True, dim_feedforward=self.d_model * self.ff_factor,
                                                   dropout=self.dropout_rate) for i in range(self.e_layers))
        self.batchNorm = nn.ModuleList(nn.BatchNorm1d(self.input_token_size) for i in range(self.e_layers))
        self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1, batch_first=True, dropout=0.5)
        self.channelBatchNorm = nn.BatchNorm1d(self.enc_in)
        self.predict = nn.Linear(self.d_model*self.input_token_size, self.pred_len)

        self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [Batch, Input length, Channel]
        # norm
        x = self.revinLayer(x_enc, 'norm').permute(0, 2, 1)

        # mapping
        inputTokens = self.mapping(x.unfold(dimension=-1, size=self.win_len, step=self.win_stride)) # b c it d

        # add position
        batch_index = torch.arange(self.input_token_size).expand(x_enc.size(0)*self.enc_in, self.input_token_size).to(x.device)
        inputTokens = inputTokens.view(-1,self.input_token_size, self.d_model)+self.positionEmbbeding(batch_index)

        # transformer extrac feature
        for i in range(self.e_layers):
            inputTokens = self.batchNorm[i](self.transformerEncoderLayers[i](inputTokens)) + inputTokens

        # mix channel if 1
        if self.channel_attn == 1:
            inputTokens = inputTokens.view(-1, self.enc_in, self.input_token_size, self.d_model)
            inputTokens = inputTokens.permute(0,2,1,3).reshape(-1,self.enc_in, self.d_model)
            inputTokens = self.channelBatchNorm(self.attn(inputTokens, inputTokens, inputTokens)[0]) + inputTokens
            inputTokens = inputTokens.view(-1, self.input_token_size, self.enc_in, self.d_model).permute(0,2,1,3)

        # predict
        y = self.predict(inputTokens.reshape(-1,self.enc_in,self.d_model*self.input_token_size)).permute(0,2,1) # b,s,c

        # denorm
        y = self.revinLayer(y, 'denorm')

        return y




