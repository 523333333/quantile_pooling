import torch
import torch.nn as nn
import torch.nn.functional as F
from quantile_pooling.quantile_pooling import quant_pooling

class DSS(nn.Module):
    def __init__(self, num_layers=3, in_channels=1, out_channels=1, d_model=64, pool="quant", single_output=False):
        super(DSS, self).__init__()

        D = d_model
        channels = [in_channels, 64, D]
        mlp = []
        tmp = in_channels
        for i in range(1, len(channels)):
            mlp += [nn.Conv1d(tmp, channels[i], 1),
                    nn.BatchNorm1d(channels[i]),
                    nn.ReLU(inplace=True)]
            tmp = channels[i]
        self.mlp = nn.Sequential(*mlp)

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if pool == "quant":
            self.aggr = quant_pooling
        elif pool == "mean":
            self.aggr = avg_pool
        elif pool=="max":
            self.aggr = max_pool

        for i in range(self.num_layers):
            self.layers.append(DSSLayer(D, D, self.aggr))

        self.dec = nn.Sequential(
            nn.BatchNorm1d(D),
            nn.ReLU(inplace=True),
            nn.Conv1d(D, D, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(D, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_channels, 1),
        )
        self.single_output = single_output

    def forward(self, input):
        """
            input: (B, N, C)
        """

        x = input.permute(0, 2, 1) # (B, C, N)
 
        x = self.mlp(x) # (B, C, N)
        B, C, N = x.shape

        for i in range(self.num_layers):
            new_x, global_feat = self.layers[i](x)
            x = x + new_x

        x = self.dec(x)
        # x: (B, C, N)
        if self.single_output:
            x = x.mean(dim=-1) # (B, C)

        return x

class DSSLayer(nn.Module):
    def __init__(self, in_C, out_C, aggr) -> None:
        super().__init__()
        self.x_bn = nn.BatchNorm1d(in_C)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_C, out_C//2, 1),
            nn.BatchNorm1d(out_C//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_C//2, out_C, 1),
        )
        self.global_feat_mlp = nn.Sequential(
            nn.Linear(in_C, out_C),
            nn.ReLU(inplace=True),
            nn.Linear(out_C, out_C),
        )
        self.aggr = aggr
        self.cat_proj = ConcatProjectLayer(out_C, out_C, out_C)

    def forward(self, x):

        x = self.x_bn(x)
        x = F.relu(x)

        x = self.mlp(x)
        
        global_feat = self.aggr(x)
        x = x - global_feat.unsqueeze(-1)

        global_feat = self.global_feat_mlp(global_feat)
        x = self.cat_proj(x, global_feat)

        return x, global_feat

    
def max_pool(x):
    return torch.max(x, -1, keepdim=False)[0] # (B, C)

def avg_pool(x):
    return torch.mean(x, -1, keepdim=False) # (B, C)

class ConcatProjectLayer(nn.Module):
    def __init__(self, A_C, B_C, out_C) -> None:
        super().__init__()

        self.project_A = nn.Conv1d(A_C, out_C, 1)
        self.project_B = nn.Linear(B_C, out_C)
        self.project_bn = nn.BatchNorm1d(out_C)

    def forward(self, x_A, x_B):
        """
            x_A: (B, C, N)
            x_B: (B, C)
        """
        x_A = self.project_A(x_A) # (B, C, N)
        x_B = self.project_B(x_B) # (B, C)
        x = x_A + x_B.unsqueeze(-1)
        x = self.project_bn(x)
        return x
    