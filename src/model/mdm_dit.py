import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./')

from model.dit import CogVideoXTransformer3DModel

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=96, dim=512):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class MDM_DiT(nn.Module):
    
    def __init__(self, n_points, n_frame, n_feats, model_config):
        super().__init__()

        self.n_points = n_points
        self.n_feats = n_feats
        self.latent_dim = model_config.latent_dim
        self.cond_seq_length = 4
        self.cond_frame = 1 if model_config.frame_cond else 0

        self.dit = CogVideoXTransformer3DModel(sample_points=n_points, sample_frames=n_frame+self.cond_frame, in_channels=n_feats,
            num_layers=model_config.n_layers, num_attention_heads=self.latent_dim // 64, cond_seq_length=self.cond_seq_length)
        
        self.input_encoder = PointEmbed(dim=self.latent_dim)
        # self.init_cond_encoder = PointEmbed(dim=self.latent_dim)
        self.E_cond_encoder = nn.Linear(1, self.latent_dim)
        self.nu_cond_encoder = nn.Linear(1, self.latent_dim)
        self.force_cond_encoder = nn.Linear(3, self.latent_dim)
        self.drag_point_encoder = nn.Linear(3, self.latent_dim)
    
    def enable_gradient_checkpointing(self):
        self.dit._set_gradient_checkpointing(True)

    def forward(self, x, timesteps, init_pc, force, E, nu, drag_mask, drag_point, floor_height=None, coeff=None, y=None, null_emb=0):
        
        """
        x: [batch_size, frame, n_points, n_feats], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        bs, n_frame, n_points, n_feats = x.shape
        
        init_pc = init_pc.reshape(bs, n_points, n_feats)
        force = force.unsqueeze(1)
        E = E.unsqueeze(1)
        nu = nu.unsqueeze(1)
        drag_point = drag_point.unsqueeze(1)
        x = torch.cat([init_pc.unsqueeze(1), x], axis=1)
        n_frame += 1
        encoder_hidden_states = torch.cat([self.force_cond_encoder(force), self.E_cond_encoder(E),
                self.nu_cond_encoder(nu), self.drag_point_encoder(drag_point)], axis=1) 
        hidden_states = self.input_encoder(x.reshape(bs * n_frame, n_points,
            n_feats)).reshape(bs, n_frame, n_points, self.latent_dim)
        full_seq = torch.cat([encoder_hidden_states, hidden_states.reshape(bs, n_frame * n_points, self.latent_dim)], axis=1)
        output = self.dit(full_seq, timesteps).reshape(bs, n_frame, n_points, 3)[:, self.cond_frame:]
        output = output + init_pc.unsqueeze(1)
            
        return output

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point_num = 512
    frame_num = 6
    
    x = torch.randn(2, frame_num, point_num, 3).to(device).to(torch.float16)
    timesteps = torch.tensor([999, 999]).int().to(device).to(torch.float16)
    init_pc = torch.randn(2, 1, point_num, 3).to(device).to(torch.float16)
    force = torch.randn(2, 3).to(device).to(torch.float16)
    E = torch.randn(2, 1).to(device).to(torch.float16)
    nu = torch.randn(2, 1).to(device).to(torch.float16)
    
    model = MDM_DiT([point_num], 3).to(device).to(torch.float16)
    output = model(x, timesteps, init_pc, force, E, nu)
    print(output.shape)
