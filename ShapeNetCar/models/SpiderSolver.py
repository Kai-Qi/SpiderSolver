import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from timeit import default_timer

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class SurfacePointClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SurfacePointClassifier, self).__init__()
        # Define the layers of the model
        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(128, 64)         
        self.fc3 = nn.Linear(64, num_classes) 
        self.relu = nn.ReLU()                 
        self.softmax = nn.Softmax(dim=1)      

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)             
        return self.softmax(x)      
    
    
class SpiderSolver_Attention_1D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,  onion_num = 15, surf_clustering_num = 4):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.surf_clustering_num = surf_clustering_num
        self.onion_num = onion_num

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        
        self.weight_surf = nn.Linear(dim, self.surf_clustering_num)
        self.weight_velo_car = nn.Linear(dim, self.onion_num * self.surf_clustering_num)

        self.weight_velo_velo = nn.Linear(dim, self.onion_num * self.surf_clustering_num)
        self.norm = nn.LayerNorm((self.onion_num + 1) * self.surf_clustering_num)
        
        for l in [self.weight_surf]:
            torch.nn.init.orthogonal_(l.weight) 
        for l in [self.weight_velo_velo]:
            torch.nn.init.orthogonal_(l.weight)  
        for l in [self.weight_velo_car]:
            torch.nn.init.orthogonal_(l.weight) 
 
        self.dropout_first_lap = nn.Dropout(dropout)
        self.to_q_first_lap = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k_first_lap = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v_first_lap = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
            
    def _process_coarse_attention(self, x, cfd_data, B, C):
        """Process Coarse attention between innermost circle points and surface points"""
        
        fx_mid = self.in_project_fx(x)
        surf_fx_mid = fx_mid[:, cfd_data.surf, :]
        velo_fx_mid = fx_mid[:, ~cfd_data.surf, :]
        
        # Extract innermost circle points
        velo_first_lap = velo_fx_mid[:, cfd_data.onion_index[0, :].bool(), :]
        first_lap_tokens = torch.cat([velo_first_lap, surf_fx_mid], axis=1)
        
        # Reshape for multi-head attention
        first_lap_tokens = first_lap_tokens.reshape(B, first_lap_tokens.shape[1], self.heads, self.dim_head)
        first_lap_tokens = first_lap_tokens.permute(0, 2, 1, 3)
        
        # Compute Coarse attention
        q_slice_token_first_lap = self.to_q_first_lap(first_lap_tokens)
        k_slice_token_first_lap = self.to_k_first_lap(first_lap_tokens)
        v_slice_token_first_lap = self.to_v_first_lap(first_lap_tokens)
        dots_first_lap = torch.matmul(q_slice_token_first_lap, k_slice_token_first_lap.transpose(-1, -2)) * self.scale
        attn_first_lap = self.softmax(dots_first_lap)
        attn_first_lap = self.dropout_first_lap(attn_first_lap)
        out_slice_token_first_lap = torch.matmul(attn_first_lap, v_slice_token_first_lap)
        
        # Reshape outputs
        velo_first_lap = out_slice_token_first_lap[:, :, 0:500, :].permute(0, 2, 1, 3)
        velo_first_lap = velo_first_lap.reshape(B, velo_first_lap.shape[1], velo_first_lap.shape[2] * velo_first_lap.shape[3])
        
        surf_fx_mid_first_lap = out_slice_token_first_lap[:, :, 500:, :].permute(0, 2, 1, 3)
        surf_fx_mid_first_lap = surf_fx_mid_first_lap.reshape(B, surf_fx_mid_first_lap.shape[1], 
                                                            surf_fx_mid_first_lap.shape[2] * surf_fx_mid_first_lap.shape[3])
        
        return velo_first_lap, surf_fx_mid_first_lap

    def _process_fine_attention(self, x, cfd_data, labels_SpectralClustering, B):
        """Process Fine attention with token generation"""
        
        fx_mid = self.in_project_fx(x)
        fx_mid_old = fx_mid.clone()
        
        # Get velocity cluster indices
        surf_cluster_result = labels_SpectralClustering
        closest_indices_on_surface = cfd_data.closest_indices_on_surface[0]
        velo_cluster_index = surf_cluster_result[closest_indices_on_surface]
        
        surf_fx_mid = fx_mid[:, cfd_data.surf, :]
        velo_fx_mid_old = fx_mid_old[:, ~cfd_data.surf, :].unsqueeze(1)
        
        # Create velocity tokens with onion indexing
        onion_index = cfd_data.onion_index
        onion_index = onion_index.unsqueeze(0).unsqueeze(3)
        velo_tokens = velo_fx_mid_old * onion_index
        velo_tokens = velo_tokens[:, :, None, :, :]
        
        surf_fx_mid_new = fx_mid_old[:, cfd_data.surf, :]
        
        # Create velocity token indices based on clustering
        velo_tokens_index = torch.zeros(1, 1, self.surf_clustering_num, 28504, 1).cuda()
        for i in range(self.surf_clustering_num):
            velo_tokens_index[0, 0, i, :, 0] = (velo_cluster_index == i)
        
        velo_tokens_index2 = torch.zeros(1, self.surf_clustering_num, 3586, 1).cuda()
        for i in range(self.surf_clustering_num):
            velo_tokens_index2[0, i, :, 0] = (labels_SpectralClustering == i)
        
        new_velo_tokens = velo_tokens * velo_tokens_index
        new_velo_tokens = new_velo_tokens.reshape(new_velo_tokens.shape[0],
                                                new_velo_tokens.shape[1] * new_velo_tokens.shape[2],
                                                new_velo_tokens.shape[3], new_velo_tokens.shape[4])
        
        new_velo_tokens = torch.mean(new_velo_tokens, axis=2)
        
        # Process surface tokens
        surf_fx_mid_new = torch.cat([surf_fx_mid_new[:, 0:16, :], surf_fx_mid_new[:, 112:, :]], dim=1)
        surf_fx_mid_new = surf_fx_mid_new[:, None, :, :]
        surf_fx_mid_new = surf_fx_mid_new * velo_tokens_index2
        surf_tokens = torch.mean(surf_fx_mid_new, axis=2)
        
        tokens = torch.cat([surf_tokens, new_velo_tokens], axis=1)
        tokens = self.norm(tokens.permute(0, 2, 1))
        tokens = tokens.permute(0, 2, 1)
        
        tokens = tokens.reshape(B, tokens.shape[1], self.heads, self.dim_head)
        tokens = tokens.permute(0, 2, 1, 3)
        
        # Compute Fine attention
        q_slice_token = self.to_q(tokens)
        k_slice_token = self.to_k(tokens)
        v_slice_token = self.to_v(tokens)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)
        
        return tokens, out_slice_token

    def _process_t2p_operation(self, x, cfd_data, B, C, velo_first_lap, surf_fx_mid_first_lap, out_slice_token):
        """Process T2P operation"""
        
        x_mid = self.in_project_x(x)
        surf_x_mid = x_mid[:, cfd_data.surf, :]
        velo_x_mid = x_mid[:, ~cfd_data.surf, :]
        
        # Process surface features
        weight_surf = self.weight_surf(surf_x_mid)
        out_slice_surf = out_slice_token[:, :, 0:self.surf_clustering_num, :].reshape(B, self.surf_clustering_num, C)
        feature_surf = weight_surf @ out_slice_surf
        
        # Process velocity features
        out_slice_velo = out_slice_token[:, :, self.surf_clustering_num:, :].permute(0, 2, 1, 3).reshape(B, self.onion_num * self.surf_clustering_num, C)
        weight_velo_velo = self.weight_velo_velo(velo_x_mid)
        feature_velo = weight_velo_velo @ out_slice_velo
        
        # Process velocity-car interaction features
        weight_velo_car = self.weight_velo_car(surf_x_mid)
        feature_velo_car = weight_velo_car @ out_slice_velo
        
        # Fuse surface features and Fine attention results
        feature_surf = (feature_surf + feature_velo_car) / 2.0 + surf_fx_mid_first_lap
        
        # Fuse velocity features with Fine attention results
        feature_velo[:, cfd_data.onion_index[0, :].bool(), :] = \
            feature_velo[:, cfd_data.onion_index[0, :].bool(), :] + velo_first_lap
        
        # Combine surface and velocity features
        new_feature = torch.zeros(x.shape[0], x.shape[1], x.shape[2]).cuda()
        new_feature[:, cfd_data.surf, :] = feature_surf
        new_feature[:, ~cfd_data.surf, :] = feature_velo
        
        new_feature = self.to_out(new_feature)
        return new_feature   
        
        
    def forward(self, x, cfd_data, labels_SpectralClustering):
        """
        Main forward pass
        Args:
            x: Input tensor [B, N, C]
            cfd_data: CFD data containing surface and onion indices
            labels_SpectralClustering: Spectral clustering labels
        Returns:
            new_feature: Processed output tensor [B, N, C]
        """
        
        # # # # # # # Main architecture of attention in SpiderSolver  # # # # # # # 
        B, N, C = x.shape
        num = cfd_data.onion_index.shape[0] * self.surf_clustering_num
        
        ### (1) Coarse attention: interaction between innermost circle point and surface point
        velo_first_lap, surf_fx_mid_first_lap = self._process_coarse_attention(x, cfd_data, B, C)
        
        ### (2) Fine attention: attention among tokens
        tokens, out_slice_token = self._process_fine_attention(x, cfd_data, labels_SpectralClustering, B)
        
        ### (3) T2P operation: From Token to Point-Wise Features
        new_feature = self._process_t2p_operation(x, cfd_data, B, C, velo_first_lap, 
                                                    surf_fx_mid_first_lap, out_slice_token)
        return new_feature


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class SpiderSolver_block(nn.Module):
    """Transformer encoder block."""
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            onion_num = 15,
            surf_clustering_num = 6,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = SpiderSolver_Attention_1D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, onion_num = onion_num, surf_clustering_num= surf_clustering_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, cfd_data, labels_SpectralClustering):
        fx_2 = self.Attn(self.ln_1(fx), cfd_data, labels_SpectralClustering)
        fx =  fx_2 + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

class SpiderSolver(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 ref=8,
                 unified_pos=False,
                 onion_num = 15,
                 surf_clustering_num = 6,
                 ):
        super(SpiderSolver, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([SpiderSolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      last_layer=(_ == n_layers - 1),
                                                      onion_num = onion_num,
                                                      surf_clustering_num = surf_clustering_num)
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos


    def forward(self, data):
        cfd_data, geom_data, labels_SpectralClustering = data
        x, fx, T = cfd_data.x, None, None
        x = x[None, :, :]
        if self.unified_pos:
            new_pos = self.get_grid(cfd_data.pos[None, :, :])
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx, cfd_data, labels_SpectralClustering)

        return fx[0]
    
    

    
    
    
