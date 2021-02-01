from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

class INFERSENT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(INFERSENT, self).__init__()
        self.opt = opt
        self.fc_dim = 512
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ctxR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.inputdim = 4 * 2 * opt.hidden_dim
        self.dense = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, opt.polarities_dim)
                )

    def forward(self, inputs):
        text_raw_indices1 = inputs[0] # batch_size x seq_len
        text_raw_indices2 = inputs[2]  # batch_size x seq_len
        ctx_len1 = torch.sum(text_raw_indices1 != 0, dim=1)
        ctx_len2 = torch.sum(text_raw_indices2 != 0, dim=1)
        ctx1 = self.embed(text_raw_indices1) # batch_size x seq_len x embed_dim
        ctx2 = self.embed(text_raw_indices2)  # batch_size x seq_len x embed_dim

        # sentence 1
        ctx_out1, (_, _) = self.ctx_lstm(ctx1, ctx_len1) #  batch_size x (ctx) seq_len x 2*hidden_dim
        emb1 = torch.max(ctx_out1, 1)[0] # max pool for LSTM outputs
        

        #sentence 2
        ctx_out2, (_, _) = self.ctxR_lstm(ctx2, ctx_len2)  # batch_size x (ctx) seq_len x 2*hidden_dim
        emb2 = torch.max(ctx_out2, 1)[0]  # max pool for LSTM outputs
        

        features =  torch.cat((emb1,emb2,torch.abs(emb1-emb2), emb1*emb2), 1)

        out = self.dense(features) # batch_size x class_dim

        return out