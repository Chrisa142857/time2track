import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py
from torch_cluster import fps
import json
import random

class TrajDataset(Dataset):
    def __init__(self, split, cfg):
        
        self.cfg = cfg
        self.dataset_path = cfg.dataset_path
        self.split = split
        self.stage = cfg.stage # 'shape' or 'deform'
        self.mode = cfg.mode # 'ae' or 'diff'
        self.repeat = cfg.repeat
        self.seed = cfg.seed 
        self.pc_size = cfg.pc_size
        self.n_sample_pro_model = cfg.n_sample_pro_model
        self.n_frames_interval = cfg.n_frames_interval
        self.n_training_frames = cfg.n_training_frames
        self.batch_size = cfg.batch_size
        self.has_gravity = cfg.get('has_gravity', False)
        self.max_num_forces = cfg.get('max_num_forces', 1)

        # if os.path.exists(os.path.join(self.dataset_path, cfg.dataset_list)):
        if os.path.exists(cfg.dataset_list):
            print(f'Loading {cfg.dataset_list}')
            with open(cfg.dataset_list, 'r') as f:
                self.split_lst = json.load(f)
        else:
            self.split_lst = [f for f in sorted(os.listdir(self.dataset_path)) if f.endswith('h5')]
        random.seed(0)
        random.shuffle(self.split_lst)
        print('Number of data:', len(self.split_lst))
        
        if cfg.overfit:
            self.split_lst = self.split_lst[:1]
        elif cfg.dataset_path.endswith('_test') or cfg.dataset_list.endswith('test.json') or cfg.dataset_list.endswith('test_list.json'):
            self.split_lst = self.split_lst[:100]
            print('Test split:', self.split_lst)
        else:
            if split == 'train':
                self.split_lst = self.split_lst[:-4]
            else:
                self.split_lst = self.split_lst[-8:]
                print('Test split:', self.split_lst)
        self.split_lst_save = self.split_lst.copy()
        self.split_lst_pcl_len = [49] * len(self.split_lst_save)
        # if not os.path.exists(os.path.join(self.dataset_path, f'info_deform_ae_{split}.json')):
        self.prepare_data_lst()
        # with open(os.path.join(self.dataset_path, f'info_deform_ae_{split}.json'), "w") as f:
        #     json.dump(self.models, f)
        #     print(f'Saved info_deform_ae_{split}.json')
        # else:
        #     self.models = json.load(open(os.path.join(self.dataset_path, f'info_deform_ae_{split}.json'), 'r'))
        #     print(f'Loaded info_deform_ae_{split}.json')
        
        print("Current stage: [bold red]{}[/bold red]".format(self.stage))
        print("Current mode: [bold red]{}[/bold red]".format(self.mode))
        print("Current split: [bold red]{}[/bold red]".format(self.split))
        print("Dataset is repeated [bold cyan]{}[/bold cyan] times".format(self.repeat))
        print("Length of split: {}".format(len(self.split_lst) if self.stage == 'shape' else len(self.models)))

    def prepare_data_lst(self): 
        self.models = []
        if self.stage == 'deform':
            if self.mode == 'ae':
                if self.split == 'train':
                    models_out, indices_out = self.random_sample_indexes(self.split_lst_save * self.repeat, self.split_lst_pcl_len * self.repeat)
                    self.models += [{"model": m, "indices": indices_out[i]} for i, m in enumerate(models_out)]
                else: # Evaluate
                    for m in self.split_lst_save:
                        for i in range(1, self.batch_size + 1):
                            self.models += [{"model": m, "indices": [i-1, i]}]
            elif self.mode == 'diff':
                # models_out, indices_out = self.subdivide_into_sequences(self.split_lst_save * self.repeat, self.split_lst_pcl_len * self.repeat)
                # self.models += [{"model": m, "start_idx": indices_out[i]} for i, m in enumerate(models_out)]
                self.models += [{"model": m, "start_idx": 0} for i, m in enumerate(self.split_lst_save)]
            else:
                raise NotImplementedError("mode not implemented")
    
    def __getitem__(self, index):
        if self.stage == 'deform':
            if self.mode == 'ae':
                return self.get_deform_ae(index)
            elif self.mode == 'diff':
                return self.get_deform_diff(index)

    def __len__(self):
        if self.stage == 'deform':
            if self.mode == 'ae':
                if self.split == 'train':
                    return sum(self.split_lst_pcl_len) * self.repeat
                else:
                    return len(self.split_lst_save) * self.batch_size # number of sequences
            elif self.mode == 'diff':
                return len(self.models)
            else:
                raise NotImplementedError("mode not implemented")
    
    def random_sample_indexes(self, models, models_len):
        n_sample_pro_model = self.n_sample_pro_model
        interval_between_frames = self.interval_between_frames
        n_selected_frames = self.n_selected_frames

        # Initialize output lists
        models_out = []
        indexes_out = []

        # Loop over each model
        for idx, model in enumerate(models):
            # For each sample per model
            for n in range(n_sample_pro_model):
                # Initialize indices list for current sample
                indexes = []

                # Select n_selected_frames number of indices
                for i in range(n_selected_frames):
                    # If first index, randomly select from range
                    if i == 0:
                        # indexes.append(np.random.randint(0, models_len[idx] - interval_between_frames))
                        indexes.append(np.random.randint(0, models_len[idx]))
                    else:
                        # For subsequent indices, select within interval_between_frames from the previous index
                        indexes.append( min(indexes[-1] + np.random.randint(0, interval_between_frames), models_len[idx]-1) )
                    
                # Append the selected indices and corresponding model to output lists
                indexes_out.append(sorted(indexes))
                models_out.append(model)
        
        return models_out, indexes_out  
    
    def get_deform_ae(self, index):
        model = self.models[index]
        model_name = model["model"]
        model_indices = model["indices"]

        model_info = {}
        model_info["model"] = model_name
        model_info["indices"] = model_indices

        model_metas = h5py.File(os.path.join(self.dataset_path, f'{model_name}'), 'r')
        model_pcls = torch.from_numpy(np.array(model_metas['x']))

        ind = np.random.default_rng(seed=self.seed).choice(model_pcls[0].shape[0], self.pc_size, replace=False)
        points_src = model_pcls[model_indices[0]][ind]
        points_tgt = model_pcls[model_indices[1]][ind]

        model_data = {}
        model_data['points_src'] = points_src.float()
        model_data['points_tgt'] = points_tgt.float()
        return model_data, model_info
    
    def get_deform_diff(self, index):
        
        model = self.models[index]
        model_name = model["model"]

        model_info = {}
        model_info["model"] = model_name
        model_info["indices"] = np.arange(self.n_training_frames)
        
        model_data = {}
        model_data['model'] = model_name
        
        model_metas = h5py.File(os.path.join(self.dataset_path, f'{model_name}'), 'r')
        model_pcls = torch.from_numpy(np.array(model_metas['x']))

        # if model_pcls[0].shape[0] > self.pc_size:
        #     ind = np.random.default_rng(seed=self.seed).choice(model_pcls[0].shape[0], self.pc_size, replace=False)
        #     points_src = model_pcls[:1]
        #     points_tgt = model_pcls[1:(self.n_training_frames*self.n_frames_interval+1):self.n_frames_interval][:, ind]
        # else: # No need to do fps in new dataset case (input is 2048 points)
        points_src = model_pcls[:1]
        points_tgt = model_pcls[1:(self.n_training_frames*self.n_frames_interval+1):self.n_frames_interval]

        if not 'drag_point' in model_metas: # Assume drag direction cross the sphere center
            drag_dir = np.array(model_metas['drag_force'])
            drag_dir = drag_dir / np.linalg.norm(drag_dir)
            drag_point = np.array([self.cfg.norm_fac, self.cfg.norm_fac, self.cfg.norm_fac]) + drag_dir
        else:
            drag_point = np.array(model_metas['drag_point'])

        if not 'floor_height' in model_metas:
            model_data['floor_height'] = torch.from_numpy(np.array(-2.4)).unsqueeze(-1).float()
        else:
            model_data['floor_height'] = (torch.from_numpy(np.array(model_metas['floor_height'])).unsqueeze(-1).float() - self.cfg.norm_fac) / 2
        model_data['drag_point'] = (torch.from_numpy(drag_point).float() - self.cfg.norm_fac) / 2
        model_data['points_src'] = (points_src.float() - self.cfg.norm_fac) / 2
        model_data['points_tgt'] = (points_tgt.float() - self.cfg.norm_fac) / 2

        model_data['vol'] = torch.from_numpy(np.array(model_metas['vol']))
        model_data['F'] = torch.from_numpy(np.array(model_metas['F']))
        model_data['F'] = model_data['F'][1:(self.n_training_frames*self.n_frames_interval+1):self.n_frames_interval]
        model_data['C'] = torch.from_numpy(np.array(model_metas['C']))
        model_data['C'] = model_data['C'][1:(self.n_training_frames*self.n_frames_interval+1):self.n_frames_interval]

        mask = torch.from_numpy(np.array(model_metas['drag_mask'])).bool()

        if 'gravity' in model_metas:
            model_data['gravity'] = torch.from_numpy(np.array(model_metas['gravity'])).long().unsqueeze(0)
        else:
            # print('no gravity in model_metas')
            model_data['gravity'] = torch.from_numpy(np.array(0)).long().unsqueeze(0)

        model_data['drag_point'] = (torch.from_numpy(drag_point).float() - self.cfg.norm_fac) / 2
        if model_data['drag_point'].ndim == 1: # For compatibility: only have one force
            model_data['drag_point'] = torch.cat([model_data['drag_point'], torch.tensor([mask.sum()]).float()], dim=0).unsqueeze(0)
        else:
            model_data['drag_point'] = torch.cat([model_data['drag_point'], mask.sum(dim=-1, keepdim=True).float()], dim=1)

        force_order = torch.randperm(self.max_num_forces) if self.split == 'train' else torch.arange(self.max_num_forces)
        mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
        # force_mask = torch.ones(self.max_num_forces, 1)
        # force_mask[:mask.shape[0]] *= 0
        # force_mask = force_mask[force_order].bool()

        if mask.shape[1] == 0:
            mask = torch.zeros(0, self.pc_size).bool()
            model_data['force'] = torch.zeros(0, 3)
            model_data['drag_point'] = torch.zeros(0, 4)
            model_data['base_drag_coeff'] = torch.zeros(self.max_num_forces, 1)
        elif not 'base_drag_coeff' in model_metas:
            vol = model_data['vol'].unsqueeze(0)
            total_volume = torch.sum(vol)
            masked_volume = torch.sum(vol * mask, dim=1)
            mean_masked_volume = masked_volume / mask.sum(dim=1)
            mask_ratio = masked_volume / total_volume
            base_drag_coeff = 9.8 * 1000 * mean_masked_volume / mask_ratio
            weighted_force = torch.from_numpy(np.array(model_metas['drag_force'])).float()
            weighted_force = weighted_force.unsqueeze(0) if weighted_force.ndim == 1 else weighted_force
            model_data['force'] = weighted_force / base_drag_coeff.unsqueeze(1)
            coeff = torch.zeros(self.max_num_forces, 1)
            coeff = coeff[force_order]
            coeff[:base_drag_coeff.shape[0]] = base_drag_coeff.unsqueeze(1)
            model_data['base_drag_coeff'] = coeff
            # model_data['weighted_force'] = weighted_force
        else:
            model_data['force'] = torch.from_numpy(np.array(model_metas['drag_force'])).float()
            model_data['base_drag_coeff'] = torch.from_numpy(np.array(model_metas['base_drag_coeff'])).float()
        
        model_data['is_mpm'] = torch.tensor(1).bool()
        if 'mat_type' in model_metas:
            model_data['mat_type'] = torch.from_numpy(np.array(model_metas['mat_type'])).long()
            if np.array(model_data['mat_type']).item() == 3: # Rigid dataset
                model_data['is_mpm'] = torch.tensor(0).bool()
        else: # temporary fix for elastic data
            model_data['mat_type'] = torch.tensor(0).long()
        
        if self.has_gravity and model_data['gravity'][0] == 1: # add gravity to force
            gravity = torch.tensor([[0, -1.0, 0]]).float()
            drag_point = (model_data['points_src'][0] * (model_data['vol'] / model_data['vol'].sum()).unsqueeze(1)).sum(axis=0) if model_data['is_mpm'] else model_data['points_src'][0].mean(axis=0)
            drag_point = torch.cat([drag_point, torch.tensor([self.pc_size]).float()]).unsqueeze(0)
            assert model_data['force'].sum() == 0, f'we are not supporting both drag and gravity now: {model_name}'
            model_data['force'] = torch.cat([model_data['force'], gravity], dim=0) if not model_data['force'].sum() == 0 else gravity
            model_data['drag_point'] = torch.cat([model_data['drag_point'], drag_point], dim=0) if not drag_point.sum() == 0 else drag_point
            mask = torch.cat([mask, torch.ones_like(mask).bool()], dim=0) if not mask.sum() == 0 else torch.ones(1, self.pc_size).bool()
        
        all_forces = torch.zeros(self.max_num_forces, 3)
        all_forces[:model_data['force'].shape[0]] = model_data['force']
        all_forces = all_forces[force_order]
        model_data['force'] = all_forces

        all_drag_points = torch.zeros(self.max_num_forces, 4)
        all_drag_points[:model_data['drag_point'].shape[0]] = model_data['drag_point']
        all_drag_points = all_drag_points[force_order]
        model_data['drag_point'] = all_drag_points

        if model_pcls[0].shape[0] > self.pc_size:
            ind = np.random.default_rng(seed=self.seed).choice(model_pcls[0].shape[0], self.pc_size, replace=False)
            model_data['points_src'] = model_data['points_src'][:, ind]
            model_data['points_tgt'] = model_data['points_tgt'][:, ind]
            mask = mask[:, ind] if mask.shape[-1] > self.pc_size else mask

        all_mask = torch.zeros(self.max_num_forces, self.pc_size).bool()
        all_mask[:mask.shape[0]] = mask
        all_mask = all_mask[force_order]

        model_data['mask'] = all_mask[..., None] # (n_forces, pc_size, 1) for compatibility
        model_data['E'] = torch.log10(torch.from_numpy(np.array(model_metas['E'])).unsqueeze(-1).float()) if np.array(model_metas['E']) > 0 else torch.zeros(1).float()
        model_data['nu'] = torch.from_numpy(np.array(model_metas['nu'])).unsqueeze(-1).float()

        return model_data, model_info