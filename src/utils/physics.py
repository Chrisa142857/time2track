import os 
import h5py
import torch
import torch.nn.functional as Fn
import numpy as np
import json

class DeformLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.device = "cuda"
        self.N = 2048
        self.I33 = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.N, 1, 1)
        self.dT = 0.0417
        self.grid_lim = 10
        self.grid_size = 125
        self.dx = self.grid_lim / self.grid_size
        self.inv_dx = 1 / self.dx
        self.density = 1000

    def forward_sequential(self, x, vol, F, C, frame_interval=2, norm_fac=5, v=None):
        
        # Denormalize x & Double dt (since we sample every 2 frames) for training
        if norm_fac > 0:
            x = x * 2 + norm_fac
        dT = self.dT * frame_interval

        loss = 0

        for bs in range(x.shape[0]):
            
            particle_mass = (self.density * vol[bs]).unsqueeze(-1).repeat(1, 3)

            start_t = 1 if frame_interval == 1 else 0
            end_t = x.shape[1] - 2
            for t in range(start_t, end_t):
                
                # Initialize
                grid_m = torch.zeros((self.grid_size, self.grid_size, self.grid_size), device=self.device)
                grid_v = torch.zeros((self.grid_size, self.grid_size, self.grid_size, 3), device=self.device)
                
                particle_x = x[bs, t]
                if v is not None:
                    particle_v = v[bs, t + 1]
                else:
                    particle_v = (x[bs, t + 2] - x[bs, t]) / (2 * dT)

                particle_F = F[bs, t].reshape(-1, 3, 3)
                particle_F_next = F[bs, t + 1].reshape(-1, 3, 3)
                particle_C = C[bs, t].reshape(-1, 3, 3)

                # P2G
                grid_pos = particle_x * self.inv_dx
                base_pos = (grid_pos - 0.5).int()
                fx = grid_pos - base_pos
                w = [0.5 * ((1.5 - fx) ** 2), 0.75 - ((fx - 1) ** 2), 0.5 * ((fx - 0.5) ** 2)]
                w = torch.stack(w, dim=2)
                dw = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
                dw = torch.stack(dw, dim=2)

                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            dpos = torch.tensor([i, j, k], device=self.device).unsqueeze(0).repeat(self.N, 1)
                            dpos = (dpos - fx) * self.dx
                            ix = base_pos[:, 0] + i
                            iy = base_pos[:, 1] + j
                            iz = base_pos[:, 2] + k
                            weight = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                            dweight = [dw[:, 0, i] * w[:, 1, j] * w[:, 2, k],
                                        w[:, 0, i] * dw[:, 1, j] * w[:, 2, k],
                                        w[:, 0, i] * w[:, 1, j] * dw[:, 2, k]]
                            dweight = torch.stack(dweight, dim=1) * self.inv_dx

                            v_in_add = weight.unsqueeze(-1) * particle_mass * (particle_v + \
                                (particle_C @ dpos.unsqueeze(-1)).squeeze(-1))
                            
                            flat_idx = ix * self.grid_size * self.grid_size + iy * self.grid_size + iz
                            flat_idx = flat_idx.long()
                            
                            grid_v = grid_v.view(-1, 3)
                            grid_v = grid_v.scatter_add(0, flat_idx.unsqueeze(-1).repeat(1, 3), v_in_add)
                            grid_v = grid_v.view(self.grid_size, self.grid_size, self.grid_size, 3)

                            grid_m = grid_m.view(-1)
                            grid_m = grid_m.scatter_add(0, flat_idx, weight * particle_mass[:, 0])
                            grid_m = grid_m.view(self.grid_size, self.grid_size, self.grid_size)

                # Grid Norm
                grid_m = torch.where(grid_m > 1e-15, grid_m, torch.ones_like(grid_m))
                grid_v = grid_v / grid_m.unsqueeze(-1)

                # G2P 
                new_F_pred = torch.zeros_like(particle_F)
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            dpos = torch.tensor([i, j, k], device=self.device).unsqueeze(0).repeat(self.N, 1).float() - fx
                            ix = base_pos[:, 0] + i
                            iy = base_pos[:, 1] + j
                            iz = base_pos[:, 2] + k
                            
                            weight = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                            dweight = [dw[:, 0, i] * w[:, 1, j] * w[:, 2, k],
                                        w[:, 0, i] * dw[:, 1, j] * w[:, 2, k],
                                        w[:, 0, i] * w[:, 1, j] * dw[:, 2, k]]
                            dweight = torch.stack(dweight, dim=1) * self.inv_dx
                            grid_v_local = grid_v[ix, iy, iz]
                            new_F_pred = new_F_pred + (grid_v_local.unsqueeze(-1) @ dweight.unsqueeze(1))

                F_pred = (self.I33 + new_F_pred * dT) @ particle_F
                loss = loss + Fn.l1_loss(F_pred, particle_F_next)
                # loss = loss + Fn.l1_loss(particle_F, particle_F_next)

        return loss / x.shape[0]

    def forward(self, x, vol, F, C, frame_interval=2, norm_fac=5, v=None):
        
        # Denormalize x & Double dt (since we sample every 2 frames) for training
        if norm_fac > 0:
            x = x * 2 + norm_fac
        dT = self.dT * frame_interval

        loss = 0

        bs = x.shape[0]
        start_t = 1 if frame_interval == 1 else 0
        end_t = x.shape[1] - 2
        M = bs * (end_t - start_t)

        # Initialize
        grid_m = torch.zeros((M, self.grid_size, self.grid_size, self.grid_size), device=self.device)
        grid_v = torch.zeros((M, self.grid_size, self.grid_size, self.grid_size, 3), device=self.device)

        particle_x = x[:, start_t:end_t].reshape(M, self.N, 3)
        # particle_x = x[:, (start_t+1):(end_t+1)].reshape(M, self.N, 3)

        if v is not None:
            # particle_v = v[:, start_t:end_t].reshape(M, self.N, 3)
            particle_v = v[:, (start_t+1):(end_t+1)].reshape(M, self.N, 3)
        else:
            particle_v = (x[:, (start_t+2):(end_t+2)] - x[:, start_t:end_t]) / (2 * dT)
        particle_v = particle_v.reshape(M, self.N, 3)

        particle_F = F[:, start_t:end_t].reshape(M, self.N, 3, 3)
        particle_F_next = F[:, (start_t+1):(end_t+1)].reshape(M, self.N, 3, 3)

        particle_C = C[:, start_t:end_t].reshape(M, self.N, 3, 3)
        # particle_C = C[:, (start_t+1):(end_t+1)].reshape(M, self.N, 3, 3)

        vol = vol.unsqueeze(1).repeat(1, end_t - start_t, 1).reshape(M, self.N)
        particle_mass = (self.density * vol).unsqueeze(-1).repeat(1, 1, 3)

        # P2G
        grid_pos = particle_x * self.inv_dx
        base_pos = (grid_pos - 0.5).int()
        fx = grid_pos - base_pos
        w = [0.5 * ((1.5 - fx) ** 2), 0.75 - ((fx - 1) ** 2), 0.5 * ((fx - 0.5) ** 2)]
        w = torch.stack(w, dim=3)
        dw = [fx - 1.5, -2 * (fx - 1), fx - 0.5]
        dw = torch.stack(dw, dim=3)

        for i in range(3):
            for j in range(3):
                for k in range(3):

                    dpos = torch.tensor([i, j, k], device=self.device).unsqueeze(0).unsqueeze(0).repeat(M, self.N, 1)
                    dpos = (dpos - fx) * self.dx
                    ix = base_pos[:, :, 0] + i
                    iy = base_pos[:, :, 1] + j
                    iz = base_pos[:, :, 2] + k

                    weight = w[:, :, 0, i] * w[:, :, 1, j] * w[:, :, 2, k]
                    dweight = [dw[:, :, 0, i] * w[:, :, 1, j] * w[:, :, 2, k],
                                w[:, :, 0, i] * dw[:, :, 1, j] * w[:, :, 2, k],
                                w[:, :, 0, i] * w[:, :, 1, j] * dw[:, :, 2, k]]
                    dweight = torch.stack(dweight, dim=2) * self.inv_dx

                    v_in_add = weight.unsqueeze(-1) * particle_mass * (particle_v + \
                        (particle_C @ dpos.unsqueeze(-1)).squeeze(-1))
                    
                    flat_idx = ix * self.grid_size * self.grid_size + iy * self.grid_size + iz
                    flat_idx = flat_idx.long()
                    
                    grid_v = grid_v.view(M, -1, 3)
                    grid_v = grid_v.scatter_add(1, flat_idx.unsqueeze(-1).repeat(1, 1, 3), v_in_add)
                    grid_v = grid_v.view(M, self.grid_size, self.grid_size, self.grid_size, 3)

                    grid_m = grid_m.view(M, -1)
                    grid_m = grid_m.scatter_add(1, flat_idx, weight * particle_mass[:, :, 0])
                    grid_m = grid_m.view(M, self.grid_size, self.grid_size, self.grid_size)
        # Grid Norm
        grid_m = torch.where(grid_m > 1e-15, grid_m, torch.ones_like(grid_m))
        grid_v = grid_v / grid_m.unsqueeze(-1)

        # G2P 
        new_F_pred = torch.zeros_like(particle_F)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):

                    dpos = torch.tensor([i, j, k], device=self.device).unsqueeze(0).unsqueeze(0).repeat(M, self.N, 1).float() - fx
                    ix = base_pos[:, :, 0] + i
                    iy = base_pos[:, :, 1] + j
                    iz = base_pos[:, :, 2] + k
                    weight = w[:, :, 0, i] * w[:, :, 1, j] * w[:, :, 2, k]
                    dweight = [dw[:, :, 0, i] * w[:, :, 1, j] * w[:, :, 2, k],
                                w[:, :, 0, i] * dw[:, :, 1, j] * w[:, :, 2, k],
                                w[:, :, 0, i] * w[:, :, 1, j] * dw[:, :, 2, k]]
                    
                    dweight = torch.stack(dweight, dim=2) * self.inv_dx
                    flat_idx = ix * self.grid_size * self.grid_size + iy * self.grid_size + iz
                    flat_idx = flat_idx.long()

                    grid_v = grid_v.view(M, -1, 3)
                    grid_v_local = grid_v.gather(1, flat_idx.unsqueeze(-1).repeat(1, 1, 3))
                    new_F_pred = new_F_pred + (grid_v_local.unsqueeze(-1) @ dweight.unsqueeze(2))

        F_pred = (self.I33 + new_F_pred * dT) @ particle_F
        loss = loss + Fn.l1_loss(F_pred, particle_F_next)
        return loss * (end_t - start_t)

def loss_momentum(x, vol, force, drag_pt_num, start_frame=1, frame_interval=2, 
    norm_fac=5, v=None, density=1000, dt=0.0417):
    
    # Denormalize x & Double dt (since we sample every 2 frames) for training
    if norm_fac > 0:
        x = x * 2 + norm_fac
    dt = dt * frame_interval
    
    loss = []
    if v is not None:
        v_curr = v[:, 1:-1]
    else:
        v_pos = x[:, 1:-1] - x[:, :-2]
        v_neg = x[:, 2:] - x[:, 1:-1]
        v_curr = (v_pos + v_neg) / (2 * dt)
    
    p_int = density * vol.unsqueeze(-1).unsqueeze(1) * v_curr
    p_int = p_int.sum(dim=2)
    dt_acc = torch.arange(1, x.shape[1] - 1, device=p_int.device, dtype=p_int.dtype) * dt
    force = force.unsqueeze(1)
    drag_pt_num = drag_pt_num.unsqueeze(1)
    dt_acc = dt_acc.unsqueeze(0).unsqueeze(-1).repeat(drag_pt_num.shape[0], 1, 3)
    p_ext = force * dt_acc * drag_pt_num
    p_ext = p_ext + start_frame * force * (dt / frame_interval) * drag_pt_num
    loss = Fn.mse_loss(p_int, p_ext)
    return loss