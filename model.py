import torch
from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule,
                     TransformerAttentionModule, SGCModule, APPNPModule, ACMGCNModule, FAGCNModule)

MODULES = {
    'GCN-MLP': [GCNModule, FeedForwardModule],
    'GAT-MLP': [GATModule, FeedForwardModule],
    'SAGE-MLP': [SAGEModule, FeedForwardModule],
    'GT-MLP': [TransformerAttentionModule, FeedForwardModule],
    'SGC-MLP': [SGCModule, FeedForwardModule],
    'APPNP-MLP': [APPNPModule, FeedForwardModule],
    'ACMGCN-MLP': [ACMGCNModule, FeedForwardModule],
    'FAGCN-MLP': [FAGCNModule, FeedForwardModule],
}

NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}

# Fundamental Model class
class Model(nn.Module):
    def __init__(self, model_name, num_layers, num_layers_1, input_dim, input_dim_1, hidden_dim, hidden_dim_1, output_dim, 
                    hidden_dim_multiplier, num_heads, normalization, dropout):
        super().__init__()

        normalization = NORMALIZATION[normalization]
        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim) if input_dim != 0 else nn.Identity()
        self.input_linear_1 = nn.Linear(in_features=input_dim_1, out_features=hidden_dim_1) if input_dim_1 != 0 else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList() if input_dim != 0 else nn.Identity()
        self.residual_modules_1 = nn.ModuleList() if input_dim_1 != 0 else nn.Identity()
        if input_dim != 0:
            for _ in range(num_layers):
                residual_module = ResidualModuleWrapper(module=MODULES[model_name][0],
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout,)
                self.residual_modules.append(residual_module)
        if input_dim_1 != 0:
            for _ in range(num_layers_1):
                residual_module_1 = ResidualModuleWrapper(module=MODULES[model_name][1],
                                                        normalization=normalization,
                                                        dim=hidden_dim_1,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout,)
                self.residual_modules_1.append(residual_module_1)
        
        self.output_normalization = normalization(hidden_dim) if input_dim != 0 else nn.Identity()
        self.output_normalization_1 = normalization(hidden_dim_1) if input_dim_1 != 0 else nn.Identity()
        if input_dim != 0 and input_dim_1 != 0:
            self.output_linear = nn.Linear(in_features=hidden_dim + hidden_dim_1, out_features=output_dim)
        elif input_dim != 0:
            self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        elif input_dim_1 != 0:
            self.output_linear = nn.Linear(in_features=hidden_dim_1, out_features=output_dim)
    
    def forward(self, graph, x, x_1):
        if x != None:
            x = self.input_linear(x)
            x = self.dropout(x)
            x = self.act(x)
            for residual_module in self.residual_modules:
                x = residual_module(graph, x)
            x = self.output_normalization(x)
        if x_1 != None:
            x_1 = self.input_linear_1(x_1)
            x_1 = self.dropout(x_1)        
            x_1 = self.act(x_1)        
            for residual_module_1 in self.residual_modules_1:
                x_1 = residual_module_1(graph, x_1)
            x_1 = self.output_normalization_1(x_1)

        if x != None and x_1 != None:
            x = self.output_linear(torch.cat((x, x_1), dim=1)).squeeze(1)
        elif x != None:
            x = self.output_linear(x).squeeze(1)
        elif x_1 != None:
            x = self.output_linear(x_1).squeeze(1)

        return x
