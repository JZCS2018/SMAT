from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

class AOA(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        self.fc_dim = 512
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ctxR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.aspR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.inputdim = 3 * 2 * opt.hidden_dim
        self.dense = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                self.dropout,
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
        interaction_mat1 = torch.matmul(ctx_out1, torch.transpose(asp_out1, 1, 2)) 
        alpha1 = F.softmax(interaction_mat1, dim=1) 
        beta1 = F.softmax(interaction_mat1, dim=2) 
        beta_avg1 = beta1.mean(dim=1, keepdim=True) 
        gamma1 = torch.matmul(alpha1, beta_avg1.transpose(1, 2)) 
        weighted_sum1 = torch.matmul(torch.transpose(ctx_out1, 1, 2), gamma1).squeeze(-1) 

        #sentence 2
        ctx_out2, (_, _) = self.ctxR_lstm(ctx2, ctx_len2)  
        emb2 = torch.max(ctx_out2, 1)[0]  
        asp_out2, (_, _) = self.aspR_lstm(asp2, asp_len2) 
        interaction_mat2 = torch.matmul(ctx_out2, torch.transpose(asp_out2, 1, 2)) 
        alpha2 = F.softmax(interaction_mat2, dim=1) 
        beta2 = F.softmax(interaction_mat2, dim=2)  
        beta_avg2 = beta2.mean(dim=1, keepdim=True) 
        gamma2 = torch.matmul(alpha2, beta_avg2.transpose(1, 2)) 
        weighted_sum2 = torch.matmul(torch.transpose(ctx_out2, 1, 2), gamma2).squeeze(-1) 

        features =  torch.cat((torch.abs(emb1-emb2), weighted_sum1, weighted_sum2), 1)

        out = self.dense(features) 

        return out
