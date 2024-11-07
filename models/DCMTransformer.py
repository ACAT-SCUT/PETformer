import torch
import torch.nn as nn
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    Derect Channel Mixing to Long Time Series Forecasting via Transformer
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

        # computer token sizes
        self.input_token_size = int((self.seq_len-self.win_len)/self.win_stride+1)
        self.output_token_size = int(self.pred_len/self.win_len)
        self.all_token_size = self.input_token_size+self.output_token_size

        # build model
        self.mapping = nn.Linear(self.win_len*self.enc_in, self.d_model)
        self.placeholder = nn.Parameter(torch.randn(self.d_model))
        self.positionEmbbeding = nn.Embedding(self.all_token_size, self.d_model)
        self.transformerEncoderLayers = nn.ModuleList(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True, dim_feedforward=self.d_model * self.ff_factor,
                                                   dropout=self.dropout_rate) for i in range(self.e_layers))
        self.batchNorm = nn.ModuleList(nn.BatchNorm1d(self.all_token_size) for i in range(self.e_layers))
        self.channelBatchNorm = nn.BatchNorm1d(self.enc_in)
        self.predict = nn.Linear(self.d_model, self.win_len*self.enc_in)

        self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=True)



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [Batch, Input length, Channel]
        # norm
        x = self.revinLayer(x_enc, 'norm').permute(0, 2, 1)

        # mapping
        inputTokens = self.mapping(x.unfold(dimension=-1, size=self.win_len, step=self.win_stride).permute(0,2,1,3)
                                   .reshape(-1,self.input_token_size,self.win_len*self.enc_in)) # b it d

        # build tokens
        allTokens = torch.cat((inputTokens, self.placeholder.repeat(x_enc.size(0), self.output_token_size, 1)), dim=1)

        # add position
        batch_index = torch.arange(self.all_token_size).expand(x_enc.size(0), self.all_token_size).to(x.device)
        allTokens = allTokens+self.positionEmbbeding(batch_index)

        # transformer extrac feature
        for i in range(self.e_layers):
            allTokens = self.batchNorm[i](self.transformerEncoderLayers[i](allTokens, src_mask=self.mask.to(x.device))) + allTokens

        # get outputTokens
        outputTokens = allTokens[:, self.input_token_size:, :]

        # predict
        y = self.predict(outputTokens).view(-1, self.enc_in,self.pred_len).permute(0,2,1) # b,s,c

        # denorm
        y = self.revinLayer(y, 'denorm')

        return y




