from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABL(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ABL, self).__init__()
        self.opt = opt
        self.fc_dim = 512
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ctxR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.aspR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.inputdim = 3 * 2 * opt.hidden_dim
        self.dense = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, opt.class_dim)
                )

    def forward(self, inputs):
        text_raw_indices1 = inputs[0] 
        aspect_indices1 = inputs[1] 
        text_raw_indices2 = inputs[2] 
        aspect_indices2 = inputs[3] 
        ctx_len1 = torch.sum(text_raw_indices1 != 0, dim=1)
        asp_len1 = torch.sum(aspect_indices1 != 0, dim=1)
        ctx_len2 = torch.sum(text_raw_indices2 != 0, dim=1)
        asp_len2 = torch.sum(aspect_indices2 != 0, dim=1)
        ctx1 = self.embed(text_raw_indices1) 
        asp1 = self.embed(aspect_indices1) 
        ctx2 = self.embed(text_raw_indices2) 
        asp2 = self.embed(aspect_indices2)  

        # sentence 1
        ctx_out1, (_, _) = self.ctx_lstm(ctx1, ctx_len1) 
        emb1 = torch.max(ctx_out1, 1)[0] 
        asp_out1, (_, _) = self.asp_lstm(asp1, asp_len1) 
        amb1 = torch.max(asp_out1, 1)[0]

        #sentence 2
        ctx_out2, (_, _) = self.ctxR_lstm(ctx2, ctx_len2) 
        emb2 = torch.max(ctx_out2, 1)[0]  
        asp_out2, (_, _) = self.aspR_lstm(asp2, asp_len2)  
        amb2 = torch.max(asp_out2, 1)[0]

        features =  torch.cat((torch.abs(emb1-emb2), amb1, amb2), 1)

        out = self.dense(features) 

        return out
