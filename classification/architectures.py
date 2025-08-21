from torch import nn
import torch
import torch.nn.init as init


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, args, num_classes, embeddings_dim):
        super().__init__()
        self.args = args
        modules = []
        
        for i in range(self.args.number_of_layers):
            modules.append(nn.Linear(embeddings_dim // (i+1), embeddings_dim // (i+2)))
            modules.append(nn.LayerNorm(embeddings_dim // (i+2)) if self.args.norm_type == 'layernorm' else nn.BatchNorm1d(embeddings_dim // (i+2)))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(self.args.dropout))
        
        modules.append(
            nn.Linear(embeddings_dim // (self.args.number_of_layers+1), num_classes)
        )
        
        self.model = nn.Sequential(*modules)
        self.model = self.model.float()
        
    def forward(self, x):
        return self.model(x)
    
class GFL(nn.Module):
    def __init__(self, input_dim_F1, input_dim_F2, gated_dim):
        super(GFL, self).__init__()

        self.WF1 = nn.Parameter(torch.Tensor(input_dim_F1, gated_dim))
        self.WF2 = nn.Parameter(torch.Tensor(input_dim_F2, gated_dim))

        init.xavier_uniform_(self.WF1)
        init.xavier_uniform_(self.WF2)

        dim_size_f = input_dim_F1 + input_dim_F2

        self.WF = nn.Parameter(torch.Tensor(dim_size_f, gated_dim))

        init.xavier_uniform_(self.WF)

    def forward(self, f1, f2):
        h_f1 = nn.functional.tanh(torch.matmul(f1, self.WF1))
        h_f2 = nn.functional.tanh(torch.matmul(f2, self.WF2))
        z_f = nn.functional.softmax(torch.matmul(torch.cat([f1, f2], dim=1), self.WF), dim=1)
        h_f = z_f * h_f1 + (1 - z_f) * h_f2
        return h_f